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

    def load_rating(self, username):
        """
        Load ratings from user_hist data
        
        Parameters
        ----------
        username : Str
        username to recommend

        Return
        ------
        user_favored_list : Array [[targetname, rating]]
        """
        session = self.cluster.connect()

        # rating_test table's scheme : user_id, rating, target_id
        rating_query = session.prepare(
            "SELECT * FROM member.rating WHERE username=?"
            )

        # Results are sorted by Cluster key(rating)
        user_favored_list = [[targetname, rating] for _, rating, targetname in session.execute(rating_query, [username])]

        return user_favored_list

    def get_similarity(self, username):
        """
        Load ratings from user_hist data
        
        Parameters
        ----------
        username : Str
        username to recommend

        Return
        ------
        similar_user : Array [[targetname, rating]]
        """
        session = self.cluster.connect()

        # rating_test table's scheme : user_id, rating, target_id
        rating_query = session.prepare(
            "SELECT * FROM member.imgsimilarity WHERE username=?"
            )

        # Results are sorted by Cluster key(rating)
        similar_user = [[targetname, similarity_score] for _, similarity_score, targetname in session.execute(rating_query, [username])]

        return similar_user