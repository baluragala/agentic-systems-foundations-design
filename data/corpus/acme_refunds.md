# Acme Cloud Refund and Cancellation Policy

This policy governs refunds for all Acme Cloud subscription plans. It applies to
orders placed on or after 1 January 2026.

## Refund Window

Customers may request a full refund within **30 days** of the order date. After
30 days, subscriptions are non-refundable for the current billing period, but
may be cancelled to prevent future charges.

Enterprise contracts are excluded from the standard 30-day window. Enterprise
refunds are governed by the individual contract's termination clause and must be
handled by the named technical account manager, not by support staff.

## Eligible Reasons

A refund request must cite one of the following reasons:

- **`billing_error`** — the customer was charged an incorrect amount, charged
  twice, or charged after a valid cancellation. Always eligible regardless of
  the 30-day window.
- **`service_outage`** — the customer experienced downtime exceeding the SLA
  commitment for their plan. Eligible if the outage is confirmed in the status
  history.
- **`not_as_described`** — the plan did not include a capability that Acme Cloud
  documentation stated it would. Eligible within the 30-day window.
- **`changed_mind`** — the customer no longer wants the subscription. Eligible
  within the 30-day window only, and only if fewer than 10,000 API calls have
  been made in the current period.

Requests citing any other reason are declined automatically and should be
escalated to a human agent for review.

## Order Status Rules

Refund eligibility also depends on the order's current status:

- `delivered` — eligible, subject to the reasons and window above.
- `shipped` — eligible; the refund is processed once the order is delivered.
- `processing` — not refundable, because no charge has been captured yet. The
  order should be **cancelled** instead, which is free and immediate.
- `cancelled` — no refund applies; no charge was captured.
- `refunded` — already refunded. A second refund is never issued. Duplicate
  requests must be escalated to a human agent.

## Processing Time

Approved refunds are returned to the original payment method within **5 to 7
business days**. Acme Cloud does not issue refunds as account credit unless the
customer explicitly requests it.

## What Support Staff May Not Do

Support staff and automated agents **must not**:

- Approve a refund for an Enterprise contract.
- Approve a refund outside the 30-day window without a `billing_error` reason.
- Issue a second refund against an order already marked `refunded`.
- Promise a specific refund date. Only the 5–7 business day range may be quoted.

Any request falling into these categories must be escalated to a human agent
with a summary of what was checked.
