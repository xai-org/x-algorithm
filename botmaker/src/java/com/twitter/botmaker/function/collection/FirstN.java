package com.twitter.botmaker.function.collection;

import java.util.List;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.FunctionFailure;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode2;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    argTypes = {
        "List<T>",
        "Long"
    },
    arguments = {
        "List<T> list",
        "Long n"
    },
    returnType = "List<T>",
    deprecated = false,
    description = "Returns the first n values of the list. if n > len(list) return list.",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "FirstN(List(\"asdf\", \"bas\", \"biz\"), 2)"
    }
)
public class FirstN extends FunctionNode2<Runtime, List<Object>, Long> {

  private static final Type TP = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.listOf(TP), Type.LONG),
      Type.listOf(TP)
  );

  public FirstN(String exprText, ImmutableList<ASTNode> children) throws SemanticCheckFailure {
    super(exprText, children);
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  public Object apply(Context<Runtime> context, List<Object> list, Long n) {
    if (list == null) {
      throw new FunctionFailure(
          this,
          context.getStackFrames(),
          new IllegalArgumentException("FirstN() called on null list"));
    }
    // Clamp negative n so subList(0, negative) cannot throw IndexOutOfBoundsException.
    long count = Math.max(0L, n == null ? 0L : n.longValue());
    return list.subList(0, (int) Math.min(count, list.size()));
  }
}
