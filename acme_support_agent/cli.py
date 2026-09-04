"""
cli.py — drive the agent from a terminal, approvals included.
=============================================================

    python -m acme_support_agent.cli "Refund ACME-1046, I changed my mind."
    python -m acme_support_agent.cli --eval
    python -m acme_support_agent.cli --audit t-abc123

Why a CLI when there is a service: because the approval round trip is much
easier to *understand* when you can see both halves in one terminal. The service
is how it is deployed; this is how you convince yourself it works.
"""
from __future__ import annotations

import argparse
import sys
import uuid


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Acme Cloud support agent")
    parser.add_argument("message", nargs="?", help="the customer's message")
    parser.add_argument("--thread", default=None, help="continue an existing thread")
    parser.add_argument("--customer", default="cli-user")
    parser.add_argument("--eval", action="store_true", help="run the evaluation suite")
    parser.add_argument("--audit", metavar="THREAD_ID", help="print a thread's audit trail")
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="approve any interrupt automatically (DEMO ONLY — defeats the gate)",
    )
    args = parser.parse_args(argv)

    from .runtime import SupportAgent

    agent = SupportAgent()

    if args.audit:
        print(agent.history(args.audit))
        return 0

    if args.eval:
        from .evaluate import evaluate
        report = evaluate(agent, thread_prefix=f"cli-{uuid.uuid4().hex[:6]}")
        print(report.show())
        return 0 if report.passed else 1

    if not args.message:
        parser.error("provide a message, or --eval / --audit")

    thread_id = args.thread or f"t-{uuid.uuid4().hex[:10]}"
    reply = agent.chat(args.message, thread_id=thread_id, customer_id=args.customer)

    if reply.needs_approval:
        print()
        print(reply.approval_request.summary())
        print()
        if args.auto_approve:
            decision, approver, note = True, "cli-auto", "auto-approved via --auto-approve"
        else:
            # Interactive approval. In production this is a reviewer in a queue
            # with real authentication — see service.py.
            answer = input("Approve this action? [y/N] ").strip().lower()
            decision = answer in ("y", "yes")
            approver = input("Your identifier: ").strip() or "cli-user"
            note = input("Note (optional): ").strip()
        reply = agent.approve(thread_id, approved=decision, approver=approver, note=note)

    print()
    print(f"[{reply.status}] thread {reply.thread_id}")
    print(f"tools: {' -> '.join(reply.tools_called) or '(none)'}")
    if reply.flags:
        print(f"flags: {reply.flags}")
    print()
    print(reply.answer or reply.error or "")
    print()
    print("-" * 60)
    print(agent.history(thread_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
