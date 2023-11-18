import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from cassandra.query import tuple_factory
from cassandra.query import ConsistencyLevel
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import WhiteListRoundRobinPolicy, DowngradingConsistencyRetryPolicy

class Cassandra:
    def __init__(self, address, port):
        profile = ExecutionProfile(
        load_balancing_policy=WhiteListRoundRobinPolicy([str(address)]),
        retry_policy=DowngradingConsistencyRetryPolicy(),
        consistency_level=ConsistencyLevel.LOCAL_QUORUM,
        serial_consistency_level=ConsistencyLevel.LOCAL_SERIAL,
        request_timeout=15,
        row_factory=tuple_factory
        )

        self.cluster = Cluster([str(address)], port=port, execution_profiles={EXEC_PROFILE_DEFAULT:profile})

    def load_all_user(self):
        """
        Load all user_id with user profile image path
        
        Return: List[user_id, image_path]
        """
        session = self.cluster.connect()
        rows = session.execute("SELECT * FROM users")
        res = [(user_row.user_id, user_row.image_path) for user_row in rows]

        return res