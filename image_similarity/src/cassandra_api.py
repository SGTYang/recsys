import os
from ssl import SSLContext, PROTOCOL_TLSv1_2 , CERT_REQUIRED

from cassandra.query import tuple_factory
from cassandra.query import ConsistencyLevel
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import WhiteListRoundRobinPolicy, DowngradingConsistencyRetryPolicy

class Cassandra:
    def __init__(self, address='cassandra.ap-northeast-2.amazonaws.com', port=9142):
        ssl_context = SSLContext(PROTOCOL_TLSv1_2)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ssl_context.load_verify_locations(os.path.join(dir_path, './sf-class2-root.crt'))
        ssl_context.verify_mode = CERT_REQUIRED

        auth_provider = PlainTextAuthProvider(
            username="keyspace_user+1-at-895636194780", 
            password="j/vlAkLrtf3zlGqu5DFWgmr0/2IvNJurhliLJ1w2bDQ=",
        )

        profile = ExecutionProfile(
        load_balancing_policy=WhiteListRoundRobinPolicy([str(address)]),
        retry_policy=DowngradingConsistencyRetryPolicy(),
        consistency_level=ConsistencyLevel.LOCAL_QUORUM,
        serial_consistency_level=ConsistencyLevel.LOCAL_SERIAL,
        request_timeout=15,
        row_factory=tuple_factory
        )

        self.cluster = Cluster(
            [str(address)], 
            port=port, 
            ssl_context=ssl_context,
            auth_provider=auth_provider,
            execution_profiles={EXEC_PROFILE_DEFAULT:profile},
            )

    def write_all_image_similarity(self):
        """
        Load all user_id with user profile image path
        
        Return: List[user_id, image_path]
        """
        res = []
        session = self.cluster.connect()
        rows = session.execute('select * from member.info')

        all_img_path = [row.photo for row in rows]

        return all_img_path