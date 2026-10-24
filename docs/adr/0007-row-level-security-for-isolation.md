# 0007. Row-level security as the isolation boundary

- Status: accepted, 2026-10-23

## Context

Tenant isolation implemented only as `WHERE tenant_id = ...` in application code fails silently the first time a query forgets the predicate.

## Decision

Enable row-level security on every table. Policies compare `tenant_id` with `current_setting('app.tenant_id', true)`, and a restrictive policy on documents and chunks checks ownership or group overlap through `app_can_read`. The runtime role is not the table owner, so the policies always apply to it. Each transaction sets the scope with `set_config(..., true)` before any other statement. Search queries also include an explicit tenant predicate.

`FORCE ROW LEVEL SECURITY` is not used: it would bind the owner role too, and the owner is used only by migrations and admin tooling that legitimately see all tenants. Letting maintenance through a forced policy would need a bypass setting that the runtime role could also set.

## Alternatives

- **Application predicates only**: one mistake leaks data.
- **A schema or database per tenant**: strong isolation, but migrations, connection management and cross-tenant operations multiply with the tenant count.

## Consequences

- With no scope set, every query returns zero rows: tested.
- Tenant admins skip document ACLs inside their tenant (`acl_bypass`); nothing skips the tenant policy.
- Local tests run as a non-superuser owner and a non-owner runtime role, because a superuser bypasses row-level security, so tests run as one could not exercise the policies at all.
