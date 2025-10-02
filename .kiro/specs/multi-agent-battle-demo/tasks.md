# Implementation Plan

- [x] 1. Set up demo project structure and dependencies
  - Create demo/ directory structure with backend/, frontend/, and shared/ modules
  - Configure pyproject.toml with demo dependency groups (fastapi, pygame, openai, etc.)
  - Set up uv dependency isolation using --group demo flags
  - Create demo-specific __init__.py files and module structure
  - Write basic project documentation for demo setup and usage
  - _Requirements: 4.5_

- [x] 2. Implement core battle data models
  - Create Agent, MapLocation, and BattleWorldState Pydantic models
  - Define AgentStatus, WeaponCondition, and LocationType enums
  - Implement battle-specific world state extensions with team management
  - Add model validation and serialization tests
  - Create utility functions for coordinate conversion and distance calculation
  - _Requirements: 3.1, 3.4_

- [x] 3. Build OpenAI structured output schemas
  - Define MoveAction, AttackAction, HealAction, RepairAction, and CommunicateAction models
  - Create AgentDecision composite model with primary_action and optional communication
  - Implement AIDecisionMaker class with OpenAI client integration
  - Add structured output parsing and error handling with fallback decisions
  - Write unit tests for schema validation and decision generation
  - _Requirements: 1.2, 3.3, 7.1_

- [x] 4. Create battle-specific observation policy
  - Implement BattleObservationPolicy with team-based filtering and fog of war
  - Add vision range constraints and enemy visibility rules
  - Implement team communication visibility (team-only message filtering)
  - Create should_observe_communication method for message isolation
  - Write tests for observation filtering accuracy and team communication privacy
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 5. Implement battle mechanics and combat system
  - Create BattleMechanics class with damage calculation, weapon degradation, and cooldowns
  - Implement CombatManager with process_attack, process_heal, and process_repair methods
  - Add forge location validation and weapon repair logic
  - Create process_communication method for team message handling
  - Write comprehensive tests for combat calculations and game rule enforcement
  - _Requirements: 3.1, 6.1, 6.2_

- [x] 6. Build Gunn integration layer
  - Create BattleOrchestrator wrapper around Gunn's Orchestrator
  - Implement world state synchronization between battle models and Gunn's WorldState
  - Add agent registration with team-based observation policies and permissions
  - Create effect processing pipeline for battle-specific effects
  - Write integration tests for Gunn orchestrator interaction
  - _Requirements: 1.1, 1.3, 5.1_

- [x] 7. Implement concurrent agent decision processing
  - Create _process_agent_decision method for parallel AI decision making
  - Implement _process_concurrent_intents for simultaneous intent submission
  - Add _decision_to_intents converter supporting action + communication intents
  - Ensure deterministic ordering for concurrent operations using agent_id sorting
  - Write tests for concurrent processing and decision consistency
  - _Requirements: 3.2, 3.4, 8.4_

- [x] 8. Build FastAPI backend server
  - Create BattleAPIServer class with REST endpoints for game control
  - Implement WebSocket handler for real-time game state updates
  - Add auto-startup game initialization (no user intervention required)
  - Create game loop with concurrent agent processing and state broadcasting
  - Add proper error handling and graceful shutdown procedures
  - _Requirements: 4.1, 4.3, 8.1_

- [x] 9. Implement effect processing and world state updates
  - Create effect handlers for AgentDamaged, AgentDied, WeaponDegraded, AgentHealed, WeaponRepaired
  - Implement TeamMessage effect processing with team-only visibility
  - Add world state update logic for health, weapon condition, and position changes
  - Create win condition detection and game status management
  - Write tests for effect processing accuracy and state consistency
  - _Requirements: 6.1, 6.2, 8.2_

- [x] 10. Build Pygame frontend renderer
  - Create BattleRenderer class with game visualization and UI components
  - Implement agent rendering with team colors, health bars, and weapon condition indicators
  - Add map location rendering for forges and other strategic points
  - Create real-time game state fetching from backend API
  - Add optional manual controls (SPACE/ESC) while maintaining auto-start functionality
  - _Requirements: 4.1, 4.4, 7.4_

- [x] 11. Implement team communication system
  - Add team message storage and retrieval in world state metadata
  - Create communication effect broadcasting with team visibility filtering
  - Implement message history display in frontend (last 10 team messages)
  - Add urgency-based message prioritization and display
  - Write tests for message isolation and team-only visibility
  - _Requirements: 2.2, 2.3, 3.5_

- [x] 12. Add comprehensive error handling and recovery
  - Implement BattleError hierarchy with specific error types
  - Create BattleErrorHandler with AI decision fallbacks and retry logic
  - Add network error recovery for frontend-backend communication
  - Implement graceful degradation for OpenAI API failures
  - Write tests for error scenarios and recovery behavior
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 13. Create game initialization and auto-start system
  - Implement automatic team creation and agent positioning
  - Add forge placement and map initialization
  - Create auto-start functionality on backend startup
  - Add game reset and restart capabilities
  - Write tests for initialization consistency and determinism
  - _Requirements: 1.1, 1.4_

- [x] 14. Implement performance monitoring and optimization
  - Add telemetry for decision making latency and concurrent processing
  - Create performance metrics for API response times and WebSocket updates
  - Implement frame rate monitoring and optimization for Pygame renderer
  - Add memory usage tracking for long-running simulations
  - Write performance tests validating real-time requirements
  - _Requirements: 8.3, 8.4_

- [ ] 15. Build comprehensive test suite
  - Create unit tests for all battle mechanics and AI decision logic
  - Implement integration tests for complete battle scenarios
  - Add end-to-end tests covering frontend-backend interaction
  - Create performance tests for concurrent agent processing
  - Write determinism tests ensuring reproducible battle outcomes
  - _Requirements: 5.2, 5.3, 8.5_

- [x] 16. Add demo documentation and examples
  - Write comprehensive README with setup and usage instructions
  - Create developer guide explaining Gunn integration patterns
  - Add code comments and docstrings for educational value
  - Create troubleshooting guide for common issues
  - Write architectural documentation explaining design decisions
  - _Requirements: 1.4, 5.1_

- [ ] 17. Implement advanced battle features
  - Add tactical positioning and cover mechanics
  - Implement coordinated team strategies and formations
  - Create dynamic difficulty adjustment based on team performance
  - Add battle statistics and performance analytics
  - Write tests for advanced tactical behaviors
  - _Requirements: 2.5, 3.5_

- [ ] 18. Create deployment and distribution setup
  - Add Docker containerization for easy deployment
  - Create deployment scripts for backend and frontend
  - Add environment configuration management
  - Implement health checks and monitoring endpoints
  - Write deployment documentation and troubleshooting guides
  - _Requirements: 4.4, 8.1_

- [ ] 19. Conduct demo validation and polish
  - Perform end-to-end testing with multiple battle scenarios
  - Validate educational value and Gunn feature demonstration
  - Add visual polish and user experience improvements
  - Create demo presentation materials and walkthroughs
  - Gather feedback and implement final improvements
  - _Requirements: 1.3, 5.4_

- [x] 20. Identify and implement Gunn library improvements
  - Document any missing Gunn functionality discovered during implementation
  - Create specs for generic Gunn enhancements (not game-specific)
  - Implement identified improvements following spec-driven development
  - Add feedback documentation for Gunn library maintainers
  - Write tests for new Gunn features and validate backward compatibility
  - _Requirements: 5.1, 5.2, 5.3_