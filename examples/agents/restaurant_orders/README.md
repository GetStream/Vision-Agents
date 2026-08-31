# Restaurant orders (inbound)

Answers the restaurant's number and takes an order from the menu in this directory.

```bash
cd examples/agents/restaurant_orders
uv sync
uv run restaurant_orders.py
```

Needs a router, a bought number attached to this customer, and phone hooks pointed at
the router: see `acceleration/README.md`.
