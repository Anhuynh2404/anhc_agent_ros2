#!/usr/bin/env python3
import functools
import rclpy
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity, SetEntityState
from pedsim_msgs.msg import AgentStates
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class AgentSpawner(Node):
    def __init__(self):
        super().__init__('agent_spawner')

        self.sub = self.create_subscription(
            AgentStates,
            'pedsim_simulator/simulated_agents',
            self.actor_poses_callback,
            qos_profile_sensor_data,
        )

        pedsim_dir = get_package_share_directory('pedsim_gazebo_plugin')
        with open(pedsim_dir + '/models/person_standing/model.sdf', 'r') as f:
            self.xml_string = f.read()

        self.spawn_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('spawn_entity service not available, waiting...')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/gazebo/set_entity_state service not available, waiting...')

        self.x_min = -12.5
        self.x_max = 12.8
        self.y_min = -10.0
        self.y_max = 13.0
        self.boundary_margin = 0.6

        self.max_ped = 300
        self.is_spawned = [False] * self.max_ped
        self.spawn_pending = [False] * self.max_ped

    def actor_poses_callback(self, actors):
        for idx, actor in enumerate(actors.agent_states):
            if idx >= self.max_ped:
                break

            actor_id = str(actor.id)
            clamped_pose = self.clamp_pose(actor.pose)

            if self.is_spawned[idx]:
                self.set_entity_state(actor_id=actor_id, model_pose=clamped_pose, idx=idx)
                continue

            if self.spawn_pending[idx]:
                continue

            self.spawn_pending[idx] = True
            spawn_future = self.spawn_entity(actor_id=actor_id, model_pose=clamped_pose)
            if spawn_future is not None:
                spawn_future.add_done_callback(
                    functools.partial(self.spawn_done_callback, actor_id, clamped_pose, idx)
                )
            else:
                self.spawn_pending[idx] = False

    def spawn_done_callback(self, actor_id, model_pose, idx, future):
        try:
            result = future.result()
            status_message = getattr(result, 'status_message', '')

            if result.success or 'already exists' in status_message.lower():
                self.is_spawned[idx] = True
                self.set_entity_state(actor_id=actor_id, model_pose=model_pose, idx=idx)
            else:
                self.is_spawned[idx] = False
                self.get_logger().warn(f'Failed to spawn [{actor_id}]: {status_message}')
        except Exception as e:
            self.is_spawned[idx] = False
            self.get_logger().warn(f'Spawn callback exception for [{actor_id}]: {e}')
        finally:
            self.spawn_pending[idx] = False

    def set_state_done_callback(self, actor_id, idx, future):
        try:
            result = future.result()
            status_message = getattr(result, 'status_message', '')

            if not result.success:
                if idx is not None:
                    self.is_spawned[idx] = False
                self.get_logger().warn(f'Failed to set state [{actor_id}]: {status_message}')
        except Exception as e:
            if idx is not None:
                self.is_spawned[idx] = False
            self.get_logger().warn(f'Set state callback exception for [{actor_id}]: {e}')

    def set_entity_state(self, actor_id, model_pose, idx=None):
        try:
            req = SetEntityState.Request()
            req.state.name = actor_id
            req.state.pose = model_pose
            req.state.reference_frame = 'world'

            future = self.set_state_client.call_async(req)
            future.add_done_callback(functools.partial(self.set_state_done_callback, actor_id, idx))
        except Exception as e:
            if idx is not None:
                self.is_spawned[idx] = False
            self.get_logger().warn(f'Failed to update agent state [{actor_id}]: {e}')

    def clamp_pose(self, pose):
        pose.position.x = min(
            max(pose.position.x, self.x_min + self.boundary_margin),
            self.x_max - self.boundary_margin,
        )
        pose.position.y = min(
            max(pose.position.y, self.y_min + self.boundary_margin),
            self.y_max - self.boundary_margin,
        )
        return pose

    def spawn_entity(self, actor_id, model_pose):
        try:
            req = SpawnEntity.Request()
            req.name = actor_id
            req.xml = self.xml_string
            req.robot_namespace = ''
            req.initial_pose = model_pose
            req.reference_frame = 'world'
            return self.spawn_client.call_async(req)
        except Exception as e:
            self.get_logger().warn(f'Failed to spawn entity [{actor_id}]: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    agent_spawner = AgentSpawner()

    try:
        rclpy.spin(agent_spawner)
    except KeyboardInterrupt:
        pass

    agent_spawner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
