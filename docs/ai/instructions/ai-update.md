
## Update a plugin

AI changes daily, so we need to stay current
Here is a guide on how to update a plugin

### 1. Check for outdated pip packages

```
uv pip upgrade --all
```

## 2. See if the tests still run

Run the test for that specific plugin

```python
uv run py.test plugins/myplugin/tests/*.py -m "integration"
```

## 3. Read the docs

Find the docs and python SDK for this project. Typically you can find these in the plugin readme.md

## 4. See if we're up to date

Evaluate if we're up to date. Is the plugin code still using the latest best practices. 
If possible upgrade to the latest best practices

## 5. Repeat running tests