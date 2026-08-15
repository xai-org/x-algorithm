Security Policy

Overview

This file provides repository-level security reporting guidance for the X For You Feed Algorithm repository (xai-org/x-algorithm).

The repository contains core source code used to determine which posts a viewer sees in the For You feed on X, including candidate retrieval, ranking, filtering, visibility-related systems, and related model and pipeline code. X's repository documentation also states that some deployment-related code and infrastructure are not included.

This file does not replace, expand, or modify the official X / xAI vulnerability disclosure or bug bounty terms. The current X / xAI Bug Bounty Program on HackerOne is authoritative for scope, eligibility, testing requirements, safe-harbor terms, bounty decisions, disclosure rules, and other program requirements.

Reporting a Security Vulnerability

Do not report sensitive security vulnerabilities in a public GitHub issue, pull request, discussion, commit message, or other public channel.

X's official security guidance directs security researchers to report possible vulnerabilities through the X / xAI Bug Bounty Program on HackerOne:

X / xAI Bug Bounty Program: https://hackerone.com/x

X Help — Reporting security vulnerabilities and bugs: https://help.x.com/en/rules-and-policies/reporting-security-vulnerabilities

X asks reports to include information and detailed instructions for reproducing the issue.

When submitting a report, include as much of the following as is safely available:

A clear description of the vulnerability and its security impact.

The affected X / xAI asset, service, repository component, file, module, or code path.

The exact conditions required to reproduce the issue.

Step-by-step reproduction instructions.

A minimal proof of concept, where appropriate and permitted by the official program rules.

Relevant request/response data, logs, stack traces, screenshots, or other evidence.

The tested environment, revision, commit hash, branch, or build information.

Whether the issue was reproduced against open-source code only, an X / xAI service, or both.

Any known prerequisites, permissions, account states, or configuration requirements.

A concise explanation of the confidentiality, integrity, authorization, or other security boundary that is affected.

Any mitigations or fixes you believe may be useful.

Do not include unnecessary personal data, authentication secrets, access tokens, credentials, private user content, or other sensitive data in a report.

Research and Testing Requirements

Follow the current X / xAI HackerOne program rules before performing security testing.

X's official Help Center specifically instructs security researchers to:

Test against accounts they control.

Use an X account with protected posts when researching security issues, to reduce the risk of publicly disclosing vulnerability details.

Do not use this repository as authorization to test systems, accounts, data, infrastructure, or services that are not permitted by the current official X / xAI bug bounty policy.

Do not intentionally access, modify, retain, destroy, or disclose data belonging to other users.

If testing could affect other users, production availability, data integrity, or service reliability, stop and follow the official HackerOne program instructions.

Repository-Specific Security Issues

The current xai-org/x-algorithm repository is intended to provide transparency into code that affects post distribution in the X For You feed. According to the repository documentation:

In-network and out-of-network posts are retrieved and ranked together.

Ranking and visibility filtering are separate parts of the system.

Visibility-related components can determine whether content is shown, dropped, or shown with a treatment.

The repository includes model, retrieval, ranking, filtering, labeling, and enforcement-related code.

Some code is designed to run end-to-end, while other areas omit deployment or infrastructure details.

Some files are intentionally not published where disclosure could increase the risk of gaming or abuse.

A security issue discovered in this repository may therefore differ from a directly exploitable vulnerability in X production.

Potentially security-relevant findings may include, subject to the current official X / xAI HackerOne scope and rules:

Unauthorized disclosure of non-public user or system information.

Authorization or access-control bypasses.

Security-boundary bypasses that expose protected or private content.

Vulnerabilities that could alter, inject, or tamper with protected data or trusted system behavior.

Remote or local code-execution vulnerabilities in runnable components.

Injection vulnerabilities.

Memory-safety vulnerabilities with a meaningful security impact.

Exposure of secrets, credentials, tokens, or sensitive configuration.

Vulnerabilities in security-sensitive filtering, labeling, enforcement, or request-handling logic where the result creates a genuine security impact.

Supply-chain or dependency vulnerabilities that are actually reachable and security-relevant in an X / xAI in-scope asset.

The presence of a bug, unusual ranking result, model error, or undesirable recommendation does not by itself establish a security vulnerability.

What Belongs on GitHub Instead

Use the repository's normal GitHub issue or pull-request workflow for non-sensitive engineering and transparency work, such as:

Reproducible non-security bugs in the open-source code.

Build or documentation problems.

Feature requests.

Code-quality improvements.

Performance improvements that do not reveal a security weakness.

Questions about how the published algorithm works.

Suggestions for improving recommendation quality.

Transparency or explainability feedback.

Model or ranking behavior that does not create a security boundary violation.

Before posting publicly, make sure the report does not contain exploit details or information that could expose a security vulnerability.

If you are unsure whether an issue is security-sensitive, prefer the private HackerOne route.

Algorithmic Abuse, Spam, Safety, and Policy Reports

Not every harmful behavior is a software security vulnerability.

Reports about spam, platform manipulation, content-policy violations, abusive accounts, or ordinary account moderation should use the appropriate X reporting and support channels unless the report demonstrates a qualifying security vulnerability covered by the official bug bounty program.

Security reports should focus on a concrete technical weakness and its reproducible security impact.

Personal Account Issues

The security vulnerability reporting process is not the support channel for a compromised or locked personal account.

If your own X account has been hacked, compromised, locked, or otherwise requires account support, use X's official Help Center and account-support processes instead of opening a repository security report.

See:

https://help.x.com/en

https://help.x.com/en/rules-and-policies/reporting-security-vulnerabilities

Supported Code and Versions

This repository is an open-source transparency repository and does not publish a repository-specific security-support lifecycle in its current documentation.

Accordingly:

Do not assume that every historical commit, branch, fork, example, synthetic dataset, or local build corresponds to production X.

Do not assume that code present in the repository is deployed exactly as published.

Do not assume that omitted deployment or infrastructure code is in scope for testing.

When reporting, identify the exact commit, branch, module, and environment you tested.

The official X / xAI HackerOne program determines whether an affected asset and vulnerability are eligible for handling or bounty consideration.

Security issues that only affect a third-party fork or modified deployment should normally be reported to that fork's maintainer unless the same issue also affects an asset covered by X / xAI's current program.

Bounties, Eligibility, and Severity

This repository does not independently promise a bounty, severity rating, response time, remediation deadline, or disclosure date.

All bounty eligibility, severity assessment, duplicate handling, reward decisions, scope determinations, testing restrictions, disclosure conditions, and related terms are governed by the current X / xAI HackerOne program policy.

Always review the live HackerOne policy before beginning testing and again before submitting a report, because scope and rules can change.

Responsible Disclosure

Keep vulnerability details private while X / xAI evaluates the report and follow the disclosure requirements of the current HackerOne program.

Do not publish exploit instructions, proof-of-concept code, screenshots, logs, reproduction steps, or other sensitive details before disclosure is permitted under the applicable program rules.

Public GitHub issues and pull requests are not appropriate for undisclosed vulnerabilities.

Security Fix Contributions

If a vulnerability requires a source-code change, report the vulnerability privately first through the official program rather than opening a public pull request that reveals the issue.

A public fix can unintentionally disclose the vulnerability before affected systems are evaluated or remediated.

Coordinate any security-sensitive code contribution or disclosure with X / xAI through the official reporting channel.

Official References

The following sources are the basis for this repository-level guidance:

X For You Feed Algorithm — official xai-org/x-algorithm repository
https://github.com/xai-org/x-algorithm

X Help — How to report security vulnerabilities
https://help.x.com/en/rules-and-policies/reporting-security-vulnerabilities

X / xAI Bug Bounty Program — HackerOne
https://hackerone.com/x

Legacy official X/Twitter Recommendation Algorithm security policy
https://github.com/twitter/the-algorithm/security/policy

The legacy twitter/the-algorithm repository explicitly directed sensitive security issues to X/Twitter's bug bounty program rather than GitHub. The current X Help Center likewise directs possible security vulnerabilities to X's security team through HackerOne.

Policy Authority

If any statement in this file conflicts with the current X / xAI HackerOne policy or an official X security instruction, the current official X / xAI policy controls.

This file should be updated when X / xAI changes its official vulnerability-reporting process, repository architecture, or security program.
