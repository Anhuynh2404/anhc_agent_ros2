#!/usr/bin/env python3
"""
pedsim_gz_bridge.py — Pedsim → Gazebo bridge node (animated-actor mode)

- Subscribes to pedsim_simulator/simulated_agents
- Spawns one Gazebo entity per pedsim agent
- Updates entity pose every tick from pedsim

Mode B (visual realism):
- Uses an animated actor visual (walking human)
- Keeps a simple collision body for lidar/physics stability
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, SetEntityState
from pedsim_msgs.msg import AgentStates


_ACTOR_SDF_TEMPLATE = """<?xml version=\"1.0\"?>
<sdf version=\"1.6\">
  <model name=\"{name}\">
    <static>false</static>
    <link name=\"base_link\">
      <inertial>
        <mass>60.0</mass>
        <inertia>
          <ixx>1.0</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>1.0</iyy><iyz>0</iyz>
          <izz>0.5</izz>
        </inertia>
      </inertial>

      <collision name=\"col\">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name=\"human_mesh\">
        <pose>0 0 {visual_z} 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>model://MaleActor/meshes/MaleVisitorWalk.dae</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>"""


class PedsimGazeboBridge(Node):
    def __init__(self):
        super().__init__("pedsim_gz_bridge")

        self.declare_parameter("world_name", "training_env_world")
        self.declare_parameter("ped_radius", 0.25)
        self.declare_parameter("ped_height", 1.70)

        self._world = self.get_parameter("world_name").value
        self._radius = float(self.get_parameter("ped_radius").value)
        self._height = float(self.get_parameter("ped_height").value)

        self._spawned: dict[str, bool] = {}
        self._spawning: set[str] = set()

        self._spawn_cli = self.create_client(SpawnEntity, "spawn_entity")
        self._state_cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")

        self.get_logger().info("Waiting for Gazebo services...")
        self._spawn_cli.wait_for_service(timeout_sec=30.0)
        self._state_cli.wait_for_service(timeout_sec=30.0)
        self.get_logger().info("Gazebo services ready.")

        self._sub = self.create_subscription(
            AgentStates,
            "pedsim_simulator/simulated_agents",
            self._on_agents,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'pedsim_gz_bridge started — world="{self._world}", '
            f"r={self._radius} m, h={self._height} m"
        )

    def _on_agents(self, msg: AgentStates):
        for agent in msg.agent_states:
            agent_id = str(agent.id)
            pose = self._make_pose(agent.pose)

            if agent_id in self._spawned:
                self._move_agent(agent_id, pose)
            elif agent_id not in self._spawning:
                self._spawn_agent(agent_id, pose)

    def _make_pose(self, pose: Pose) -> Pose:
        p = Pose()
        p.position.x = pose.position.x
        p.position.y = pose.position.y
        p.position.z = self._height / 2.0
        p.orientation = pose.orientation
        return p

    def _spawn_agent(self, agent_id: str, pose: Pose):
        visual_z = -(self._height / 2.0)
        sdf = _ACTOR_SDF_TEMPLATE.format(
            name=f"ped_{agent_id}",
            radius=self._radius,
            height=self._height,
            visual_z=visual_z,
        )

        req = SpawnEntity.Request()
        req.name = f"ped_{agent_id}"
        req.xml = sdf
        req.robot_namespace = ""
        req.initial_pose = pose
        req.reference_frame = "world"

        self._spawning.add(agent_id)
        future = self._spawn_cli.call_async(req)
        future.add_done_callback(lambda f: self._spawn_done(f, agent_id))

    def _spawn_done(self, future, agent_id: str):
        self._spawning.discard(agent_id)
        try:
            result = future.result()
            if result.success:
                self._spawned[agent_id] = True
                self.get_logger().info(f'Spawned animated pedestrian "ped_{agent_id}"')
            else:
                self.get_logger().warn(
                    f'Failed to spawn "ped_{agent_id}": {result.status_message}'
                )
                self._spawned.pop(agent_id, None)
        except Exception as e:
            self.get_logger().error(f'Spawn service exception for "ped_{agent_id}": {e}')
            self._spawned.pop(agent_id, None)

    def _move_agent(self, agent_id: str, pose: Pose):
        req = SetEntityState.Request()
        req.state.name = f"ped_{agent_id}"
        req.state.pose = pose
        req.state.reference_frame = "world"
        self._state_cli.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = PedsimGazeboBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
