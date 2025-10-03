"""Demo script for Web adapter functionality.

This script demonstrates how to set up and use the Web adapter for external
system integration with REST and WebSocket endpoints.
"""

import asyncio
import json
import socket
import time
from threading import Thread

import httpx
import uvicorn
import websockets

from gunn.adapters.web import AuthToken, WebAdapter, create_web_adapter
from gunn.core.orchestrator import AgentHandle, Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect, ObservationDelta


class DemoObservationPolicy(ObservationPolicy):
    """Demo observation policy for the example."""

    def __init__(self):
        from gunn.policies.observation import PolicyConfig

        config = PolicyConfig()
        super().__init__(config)

    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Return a filtered view for the agent."""
        return View(
            agent_id=agent_id,
            view_seq=0,
            visible_entities=world_state.entities,
            visible_relationships=world_state.relationships,
            context_digest=f"digest_{agent_id}_{int(time.time())}",
        )

    def should_observe_event(
        self, effect: Effect, agent_id: str, world_state: WorldState
    ) -> bool:
        """Determine if agent should observe this event."""
        # For demo, agents observe all events
        return True

    def calculate_observation_delta(
        self, old_view: View, new_view: View
    ) -> ObservationDelta:
        """Calculate observation delta between views."""
        return ObservationDelta(
            view_seq=new_view.view_seq + 1,
            patches=[
                {
                    "op": "replace",
                    "path": "/entities/demo",
                    "value": {"updated_at": time.time()},
                }
            ],
            context_digest=new_view.context_digest,
            schema_version="1.0.0",
        )


async def setup_demo_environment() -> tuple[
    Orchestrator, WebAdapter, AgentHandle, AgentHandle
]:
    """Set up the demo environment with orchestrator and agents."""
    print("Setting up demo environment...")

    # Create orchestrator
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="demo_world")
    await orchestrator.initialize()

    # Register demo agents
    policy = DemoObservationPolicy()
    agent1 = await orchestrator.register_agent("alice", policy)
    agent2 = await orchestrator.register_agent("bob", policy)

    print("Registered agents: alice, bob")

    # Create authentication tokens
    auth_tokens: dict[str, AuthToken] = {
        "alice_token": AuthToken(
            token="alice_token",
            world_id="demo_world",
            agent_id="alice",
            permissions=["submit_intent", "get_observation", "stream_observations"],
        ),
        "bob_token": AuthToken(
            token="bob_token",
            world_id="demo_world",
            agent_id="bob",
            permissions=["submit_intent", "get_observation", "stream_observations"],
        ),
    }

    # Create web adapter
    web_adapter = create_web_adapter(
        orchestrator=orchestrator,
        auth_tokens=auth_tokens,
        rate_limit_requests=100,
        rate_limit_window=60,
    )

    print("Web adapter created with authentication tokens")

    return orchestrator, web_adapter, agent1, agent2


async def demo_rest_api():
    """Demonstrate REST API functionality."""
    print("\n=== REST API Demo ===")

    # Find available port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    _orchestrator, web_adapter, _agent1, _agent2 = await setup_demo_environment()

    # Start server in thread
    server_thread = Thread(
        target=uvicorn.run,
        args=(web_adapter.app,),
        kwargs={"host": "127.0.0.1", "port": port, "log_level": "error"},
        daemon=True,
    )
    server_thread.start()

    # Wait for server to start
    await asyncio.sleep(1.0)

    base_url = f"http://127.0.0.1:{port}"

    async with httpx.AsyncClient() as client:
        # Test health check
        print("Testing health check...")
        response = await client.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")

        # Test intent submission
        print("\nTesting intent submission...")
        headers = {"Authorization": "Bearer alice_token"}
        intent_data = {
            "kind": "Speak",
            "payload": {"text": "Hello from Alice!", "target": "bob"},
            "priority": 1,
            "context_seq": 0,
        }

        response = await client.post(
            f"{base_url}/worlds/demo_world/agents/alice/intents",
            json=intent_data,
            headers=headers,
        )
        print(f"Intent submission: {response.status_code} - {response.json()}")

        # Test unauthorized access
        print("\nTesting unauthorized access...")
        response = await client.post(
            f"{base_url}/worlds/demo_world/agents/alice/intents",
            json=intent_data,
            # No authorization header
        )
        print(f"Unauthorized access: {response.status_code}")

        # Test wrong agent access
        print("\nTesting wrong agent access...")
        response = await client.post(
            f"{base_url}/worlds/demo_world/agents/bob/intents",  # Wrong agent
            json=intent_data,
            headers=headers,  # Alice's token
        )
        print(f"Wrong agent access: {response.status_code}")

        # Test observation retrieval (with timeout)
        print("\nTesting observation retrieval...")
        response = await client.get(
            f"{base_url}/worlds/demo_world/agents/alice/observations?timeout=0.5",
            headers=headers,
        )
        print(f"Observation retrieval: {response.status_code}")
        if response.status_code == 408:
            print("  (Timeout expected - no observations queued)")

    await web_adapter.shutdown()
    print("REST API demo completed")


async def demo_websocket() -> None:
    """Demonstrate WebSocket functionality."""
    print("\n=== WebSocket Demo ===")

    # Start server in background
    import socket
    from threading import Thread

    import uvicorn

    # Find available port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    orchestrator, web_adapter, _agent1, _agent2 = await setup_demo_environment()

    # Start server in thread
    server_thread = Thread(
        target=uvicorn.run,
        args=(web_adapter.app,),
        kwargs={"host": "127.0.0.1", "port": port, "log_level": "error"},
        daemon=True,
    )
    server_thread.start()

    # Wait for server to start
    await asyncio.sleep(1.0)

    # Test WebSocket connection
    uri = f"ws://127.0.0.1:{port}/worlds/demo_world/agents/alice/observations/stream?token=alice_token"

    try:
        print("Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("WebSocket connected successfully")

            # Send some events to trigger observations
            print("Broadcasting events...")
            await orchestrator.broadcast_event(
                {
                    "kind": "MessageSent",
                    "payload": {"from": "bob", "to": "alice", "text": "Hello Alice!"},
                    "source_id": "bob",
                    "schema_version": "1.0.0",
                }
            )

            await orchestrator.broadcast_event(
                {
                    "kind": "UserJoined",
                    "payload": {"user_id": "charlie", "room": "demo_room"},
                    "source_id": "system",
                    "schema_version": "1.0.0",
                }
            )

            # Try to receive observations
            print("Waiting for observations...")
            try:
                for i in range(3):
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    print(f"Received observation {i + 1}: {data}")
            except TimeoutError:
                print("No observations received (this is expected in demo)")

    except Exception as e:
        print(f"WebSocket demo failed: {e}")
        print("(This is expected in test environment)")

    await web_adapter.shutdown()
    print("WebSocket demo completed")


async def demo_authentication() -> None:
    """Demonstrate authentication and authorization features."""
    print("\n=== Authentication Demo ===")

    _orchestrator, web_adapter, _agent1, _agent2 = await setup_demo_environment()

    # Add a token with limited permissions
    web_adapter.add_auth_token(
        token="limited_token",
        world_id="demo_world",
        agent_id="alice",
        permissions=["get_observation"],  # No submit_intent permission
    )

    # Add an expired token
    web_adapter.add_auth_token(
        token="expired_token",
        world_id="demo_world",
        agent_id="alice",
        permissions=["submit_intent", "get_observation"],
        expires_at=time.time() - 3600,  # Expired 1 hour ago
    )

    print("Added tokens:")
    print("- limited_token: only get_observation permission")
    print("- expired_token: expired 1 hour ago")

    # Test token validation
    tokens_to_test = [
        ("alice_token", "Valid token"),
        ("limited_token", "Limited permissions"),
        ("expired_token", "Expired token"),
        ("invalid_token", "Invalid token"),
    ]

    for token, description in tokens_to_test:
        print(f"\nTesting {description} ({token}):")

        if token in web_adapter.auth_tokens:
            auth_token = web_adapter.auth_tokens[token]
            print(f"  World ID: {auth_token.world_id}")
            print(f"  Agent ID: {auth_token.agent_id}")
            print(f"  Permissions: {auth_token.permissions}")

            if auth_token.expires_at:
                expired = time.time() > auth_token.expires_at
                print(f"  Expired: {expired}")
        else:
            print("  Token not found")

    # Remove tokens
    web_adapter.remove_auth_token("limited_token")
    web_adapter.remove_auth_token("expired_token")
    print("\nRemoved test tokens")

    await web_adapter.shutdown()
    print("Authentication demo completed")


async def main() -> None:
    """Run all demos."""
    print("Web Adapter Demo")
    print("================")

    try:
        await demo_rest_api()
        await demo_websocket()
        await demo_authentication()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()

    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())
