# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

DWELL_INDEX = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    p.add_argument("--data", required=True, help="dump dir from reference/dump_gen.py")
    p.add_argument("--sessions", type=int, default=3, help="user sessions to drive")
    p.add_argument("--topk", type=int, default=16, help="candidates to retrieve per user")
    p.add_argument("--retrieval-port", type=int, default=9990)
    p.add_argument("--ranking-port", type=int, default=9988)
    p.add_argument("--timeout-s", type=float, default=60.0)
    return p.parse_args(argv)


def _load_world_tables(data_dir: Path):
    import pyarrow.parquet as pq

    posts = pq.read_table(data_dir / "world" / "world_posts.parquet")
    post_text = dict(zip(posts.column("post_id").to_pylist(), posts.column("text").to_pylist()))
    users = pq.read_table(data_dir / "world" / "world_users.parquet")
    user_handle = dict(zip(users.column("user_id").to_pylist(), users.column("handle").to_pylist()))
    return post_text, user_handle


def _iter_sessions(data_dir: Path, n: int):
    import pyarrow.parquet as pq

    sys.path.insert(0, str(HERE))
    from dump_gen import _batch_relpath

    table = pq.read_table(data_dir / _batch_relpath(0, 0)).slice(0, n)
    for i in range(min(n, table.num_rows)):
        padding = np.array(table.column("paddingMask")[i].as_py(), dtype=bool)
        new_event = np.array(table.column("newEventMask")[i].as_py(), dtype=bool)
        hist = np.flatnonzero(padding & ~new_event)
        yield {
            "user_id": table.column("userId")[i].as_py(),
            "hist_idx": hist,
            "tweet": np.array(table.column("tweetIdSeq")[i].as_py()),
            "author": np.array(table.column("authorIdSeq")[i].as_py()),
            "impressed": np.array(table.column("impressedTimeMsSeq")[i].as_py()),
            "actions": np.array(table.column("actionNameMultiHotSeqSeq")[i].as_py()),
            "dwell": np.array(table.column("continuousActionValuesSeqSeq")[i].as_py())[
                :, DWELL_INDEX
            ],
        }


def _session_to_user_action_sequence(sess) -> object:
    from xai_proto import recsys_pb2

    uid = int(sess["user_id"])
    aggregated = []
    for j in sess["hist_idx"]:
        action_infos = [
            recsys_pb2.ActionInfo(
                engagementTimeMs=int(sess["impressed"][j]),
                actionName=int(a),
                userActionMeta=recsys_pb2.UserActionMeta(
                    indexInSequence=int(j),
                    page="home",
                    dwellTime=float(sess["dwell"][j]),
                ),
            )
            for a in np.flatnonzero(sess["actions"][j])
        ]
        aggregated.append(
            recsys_pb2.AggregatedUserAction(
                userId=uid,
                tweetInfo=recsys_pb2.TweetInfo(
                    tweetId=int(sess["tweet"][j]), authorId=int(sess["author"][j])
                ),
                actions=action_infos,
            )
        )
    times = sess["impressed"][sess["hist_idx"]]
    return recsys_pb2.UserActionSequence(
        userId=uid,
        userActionsData=recsys_pb2.UserActionSequenceDataContainer(
            orderedAggregatedUserActionsList=recsys_pb2.AggregatedUserActionList(
                aggregatedUserActions=aggregated
            ),
        ),
        metadata=recsys_pb2.UserActionSequenceMeta(
            length=len(aggregated),
            firstSequenceTime=int(times[0]) if len(times) else 0,
            lastSequenceTime=int(times[-1]) if len(times) else 0,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data).expanduser()
    if not data_dir.is_dir():
        raise SystemExit(f"error: --data directory does not exist: {data_dir}")

    import grpc
    from xai_proto import recsys_pb2, recsys_pb2_grpc

    post_text, user_handle = _load_world_tables(data_dir)
    retrieval = recsys_pb2_grpc.RecsysRetrievalPredictorStub(
        grpc.insecure_channel(f"localhost:{args.retrieval_port}")
    )
    ranking = recsys_pb2_grpc.RecsysPredictorStub(
        grpc.insecure_channel(f"localhost:{args.ranking_port}")
    )

    n_ok = 0
    for sess in _iter_sessions(data_dir, args.sessions):
        uid = int(sess["user_id"])
        uas = _session_to_user_action_sequence(sess)

        r_reply = retrieval.RetrieveTopKCandidates(
            recsys_pb2.RetrieveTopKCandidatesRequest(
                user_ids=[uid],
                sequences=[uas],
                topK=args.topk,
                metadata="retrieve_then_rank",
            ),
            timeout=args.timeout_s,
        )
        retrieved = list(r_reply.topKCandidates[0].candidates)
        if not retrieved:
            print(f"user {uid}: retrieval returned no candidates", flush=True)
            continue
        retrieval_score = {c.candidate.tweetId: c.score for c in retrieved}

        cand_set = recsys_pb2.CandidateSet(
            userId=uid,
            candidates=[
                recsys_pb2.TweetInfo(tweetId=c.candidate.tweetId, authorId=c.candidate.authorId)
                for c in retrieved
            ],
        )
        k_reply = ranking.PredictNextActions(
            recsys_pb2.PredictNextActionsRequest(sequences=[uas], candidateSets=[cand_set]),
            timeout=args.timeout_s,
        )
        dists = k_reply.distributionSets[0].candidateDistributions
        fav_idx = recsys_pb2.ActionName.Value("SERVER_TWEET_FAV")
        rows = []
        for c, dist in zip(retrieved, dists):
            logits = list(dist.logits)
            p_fav = (
                1.0 / (1.0 + np.exp(-logits[fav_idx])) if fav_idx < len(logits) else float("nan")
            )
            rows.append((c.candidate.tweetId, p_fav))
        rows.sort(key=lambda r: -r[1])

        handle = user_handle.get(uid, f"user_{uid}")
        print(f"-- {handle}  (history={len(sess['hist_idx'])}, retrieved={len(retrieved)})")
        for rank, (pid, p_fav) in enumerate(rows, 1):
            text = post_text.get(pid, f"post_{pid}")
            print(
                f"   {rank:>2}. P(fav)={p_fav:.4f}  "
                f"retrieval_score={retrieval_score.get(pid, float('nan')):+.4f}  {text}"
            )
        n_ok += 1

    if n_ok == 0:
        raise SystemExit("error: no session completed the retrieve->rank loop")
    print(f"retrieve_then_rank: {n_ok} session(s) completed the full loop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
