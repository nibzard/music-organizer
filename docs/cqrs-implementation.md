# CQRS Implementation in Music Organizer

This document describes the Command Query Responsibility Segregation (CQRS) pattern implementation in the Music Organizer application.

## Overview

CQRS is a pattern that separates read (query) and write (command) operations for a data store. This separation provides several benefits:

- **Scalability**: Read and write operations can scale independently
- **Performance**: Optimized read models for different query patterns
- **Maintainability**: Clear separation of concerns between commands and queries
- **Flexibility**: Easy to add new queries without affecting write operations

## Architecture

### Core Components

#### 1. Commands (Write Model)
Commands represent intentions to change the system state. They are:

- Immutable data structures containing all necessary data
- Processed by specific command handlers
- Result in domain events that describe what happened
- Always return a result indicating success/failure

```python
@dataclass(frozen=True, slots=True)
class AddRecordingCommand(Command):
    """Command to add a new recording to the catalog."""
    file_path: Path
    metadata: Dict[str, Any]
    catalog_id: str = "default"
```

#### 2. Queries (Read Model)
Queries represent requests for information. They are:

- Immutable data structures specifying what data is needed
- Processed by specific query handlers
- Never modify system state
- Support caching for performance optimization

```python
@dataclass(frozen=True, slots=True)
class GetRecordingByIdQuery(Query):
    """Query to get a recording by ID."""
    recording_id: str
```

#### 3. Events (Domain Events)
Events represent facts about things that have happened in the system:

- Immutable records of past occurrences
- Published when commands are successfully executed
- Used to update read models and notify other systems
- Can be stored for event sourcing

```python
class RecordingAddedEvent(DomainEvent):
    """Event raised when a recording is added to the catalog."""

    def __init__(self, recording_id: str, file_path: Path, catalog_id: str):
        super().__init__(
            aggregate_id=recording_id,
            aggregate_type="Recording",
            event_type="RecordingAdded",
            event_data={
                "file_path": str(file_path),
                "catalog_id": catalog_id
            }
        )
```

#### 4. Buses (Mediators)

**CommandBus**: Routes commands to appropriate handlers
- Maintains a registry of command types and their handlers
- Supports middleware for cross-cutting concerns (logging, validation)
- Handles errors and returns command results

**QueryBus**: Routes queries to appropriate handlers
- Includes built-in caching support
- Supports middleware for optimization
- Returns query results with metadata

**EventBus**: Publishes events to subscribers
- Decouples publishers from subscribers
- Supports multiple handlers per event type
- Provides error isolation

### Directory Structure

```
src/music_organizer/application/
├── commands/               # Write model
│   ├── base.py            # Base command classes
│   ├── catalog/           # Catalog commands
│   ├── organization/      # Organization commands
│   └── classification/    # Classification commands
├── queries/               # Read model
│   ├── base.py            # Base query classes
│   ├── catalog/           # Catalog queries
│   ├── organization/      # Organization queries
│   └── classification/    # Classification queries
├── events/                # Domain events
│   └── base.py            # Base event classes
└── read_models/           # Denormalized read models
    └── projector.py       # Event-to-read-model projection
```

## Implementation Examples

### Adding a Recording (Command)

```python
# Create command
command = AddRecordingCommand(
    file_path=Path("/music/artist/album/track.flac"),
    metadata={
        "title": "Song Title",
        "artists": ["Artist Name"],
        "year": 2023
    }
)

# Dispatch via command bus
result = await command_bus.dispatch(command)

if result.success:
    print(f"Added recording with ID: {result.result_data['recording_id']}")
    # Events are automatically published
```

### Querying Recordings (Query)

```python
# Create query
query = GetRecordingsByArtistQuery(
    artist_name="Artist Name",
    limit=10
)

# Dispatch via query bus
result = await query_bus.dispatch(query)

if result.success:
    print(f"Found {len(result.data)} recordings")
    # Results may come from cache
```

### Handling Events

```python
class StatisticsUpdater:
    async def handle(self, event: DomainEvent):
        if event.event_type == "RecordingAdded":
            await self.update_statistics(event.event_data)

# Subscribe to events
event_bus.subscribe("RecordingAdded", StatisticsUpdater())
```

## Benefits Achieved

### 1. Separation of Concerns
- Commands focus on business logic and validation
- Queries focus on data retrieval and presentation
- Events capture the intent and outcome of operations

### 2. Performance Optimization
- Query results are cached automatically
- Read models can be optimized for specific query patterns
- No transaction conflicts between reads and writes

### 3. Extensibility
- New commands can be added without affecting queries
- New queries can be added without affecting commands
- Event handlers can be added without modifying existing code

### 4. Testability
- Commands and queries can be tested independently
- Mock implementations are easy to create
- Event-driven behavior can be tested via event store

## Usage in the Application

### Core Service Integration

```python
class MusicLibraryService:
    def __init__(self):
        self.command_bus = CommandBus()
        self.query_bus = QueryBus()
        self.event_bus = EventBus()

        # Register handlers
        self._register_handlers()

    async def add_recording(self, path: Path, metadata: dict) -> str:
        command = AddRecordingCommand(file_path=path, metadata=metadata)
        result = await self.command_bus.dispatch(command)
        return result.result_data["recording_id"]

    async def search(self, term: str) -> List[Recording]:
        query = SearchRecordingsQuery(search_term=term)
        result = await self.query_bus.dispatch(query)
        return result.data.recordings
```

### Middleware Examples

**Logging Middleware:**
```python
async def logging_middleware(next_handler):
    async def handler(command_or_query):
        print(f"Processing: {type(command_or_query).__name__}")
        result = await next_handler(command_or_query)
        print(f"Result: {result.success}")
        return result
    return handler

command_bus.register_middleware(logging_middleware)
```

**Validation Middleware:**
```python
async def validation_middleware(next_handler):
    async def handler(command):
        # Validate command before processing
        validate_command(command)
        return await next_handler(command)
    return handler
```

## Best Practices

### Commands
- Use descriptive, intent-revealing names (e.g., `AddRecordingCommand`)
- Include all necessary data in the command
- Never have optional parameters that affect behavior
- Return results with clear success/failure indication

### Queries
- Design queries around specific use cases
- Include pagination parameters for list queries
- Use descriptive cache keys for frequently executed queries
- Return DTOs optimized for presentation

### Events
- Use past tense in event names (e.g., `RecordingAdded`)
- Include all relevant data in the event
- Make events immutable
- Design events to be self-contained

### Testing
- Test command handlers in isolation with mock repositories
- Test query handlers with test data
- Test event publishing and handling separately
- Use the event store for integration testing

## Migration Strategy

The existing codebase can be gradually migrated to CQRS:

1. **Phase 1**: Extract commands from existing service methods
2. **Phase 2**: Create query handlers for read operations
3. **Phase 3**: Introduce events for state changes
4. **Phase 4**: Add read model projections
5. **Phase 5**: Optimize queries with caching and denormalization

## Performance Considerations

- **Caching**: Query results are cached with configurable TTL
- **Batch Operations**: Commands support batch processing for efficiency
- **Async Processing**: All operations are async for better I/O handling
- **Memory Efficiency**: Dataclasses with slots minimize memory usage

## Conclusion

The CQRS implementation provides a solid foundation for scaling the Music Organizer application while maintaining clean architecture and separation of concerns. The pattern enables independent optimization of read and write operations, supports event-driven architectures, and improves overall maintainability.