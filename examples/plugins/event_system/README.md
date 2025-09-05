# GetStream AI Plugins Event System Examples

This directory contains examples demonstrating how to use the GetStream AI Plugins Event System.

## Examples

### `event_system_example.py`

A comprehensive example showing:
- How to implement plugins with structured events
- Event filtering and analysis
- Performance monitoring
- Event serialization
- Error handling

To run the example:

```bash
cd /Users/dev/PycharmProjects/stream-py
python -m getstream.plugins.common.examples.event_system_example
```

## Key Features Demonstrated

1. **Structured Events**: Type-safe event classes with rich metadata
2. **Event Registry**: Global event tracking and analysis
3. **Performance Metrics**: Automatic calculation of plugin performance
4. **Event Filtering**: Flexible filtering by type, time, confidence, etc.
5. **Serialization**: Save and restore events for analysis
6. **Backward Compatibility**: Support for legacy event handlers

## Migration Examples

The examples show how to migrate from legacy event handling to the new structured system while maintaining backward compatibility.

## Best Practices

- Always use structured events for new code
- Include rich metadata in events
- Use event filtering for efficient processing
- Monitor performance with built-in metrics
- Handle errors with structured error events
