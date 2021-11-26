import peewee

db = peewee.MySQLDatabase(None)


def set_connection_parameter(config):
    """
    Sets the database connection for this peewee model
    Args:
        config: dict with elements database, host, port, user, password

    Returns:

    """
    db.init(
        database=config["database"],
        host=config["host"],
        port=config["port"],
        user=config["user"],
        passwd=config["password"]
    )


class BaseModel(peewee.Model):
    """
    Set peewee BaseModel with correct database connection.
    """

    class Meta:
        database = db


class ArrivalTimeGenerator(BaseModel):
    atgen_id = peewee.AutoField()
    name = peewee.CharField()
    parameter = peewee.TextField()  # Maybe a little bit ugly, but store the parameters as JSON.


class FlowVolumeGenerator(BaseModel):
    fvgen_id = peewee.AutoField()
    name = peewee.CharField()
    parameter = peewee.TextField()


class ConnectionPairGenerator(BaseModel):
    cpgen_id = peewee.AutoField()
    name = peewee.CharField()
    parameter = peewee.TextField()


class FlowGenerator(BaseModel):
    flowgen_id = peewee.AutoField()
    name = peewee.CharField()

    # For randomflowgenerator
    seed = peewee.IntegerField(null=True, default=None)
    number_flows = peewee.IntegerField(null=True, default=None)
    nodes = peewee.CharField(null=True, default=None)  # The group of nodes to use

    fk_arrival_time_generator = peewee.ForeignKeyField(ArrivalTimeGenerator, backref="flow_generators", null=True,
                                                       default=None)
    fk_flow_volume_generator = peewee.ForeignKeyField(FlowVolumeGenerator, backref="flow_generators", null=True,
                                                      default=None)
    fk_connection_pair_generator = peewee.ForeignKeyField(ConnectionPairGenerator, backref="flow_generators", null=True,
                                                          default=None)

    bidirectional = peewee.BooleanField(null=True, default=False)
    volume_per_node = peewee.DecimalField(max_digits=18, decimal_places=6, null=True, default=None)
    parameter = peewee.TextField(null=True, default=None)  # Other parameters


class Algorithm(BaseModel):
    algorithm_id = peewee.AutoField()
    name = peewee.CharField()
    parameter = peewee.TextField()


class Topology(BaseModel):
    topology_id = peewee.AutoField()
    name = peewee.CharField()
    parameter = peewee.TextField()

    fk_decorated_topology = peewee.ForeignKeyField('self', null=True, backref='decorators')


class Simulation(BaseModel):
    simulation_id = peewee.AutoField()
    fk_topology = peewee.ForeignKeyField(Topology, backref="simulations")
    fk_flow_generator = peewee.ForeignKeyField(FlowGenerator, backref="simulations")
    fk_algorithm = peewee.ForeignKeyField(Algorithm, backref="simulations")
    finished = peewee.BooleanField(default=False)
    duration = peewee.IntegerField(default=None)
    start_time = peewee.TimestampField(default=0)
    end_time = peewee.TimestampField(default=0)
    sha = peewee.CharField(default="None")
    simulation_behavior_name = peewee.CharField(default="None", null=True)
    simulation_builder_name = peewee.CharField(default="None", null=True)
    simulation_behavior_params = peewee.TextField(default="None", null=True)


class Flow(BaseModel):
    flow_id = peewee.BigAutoField(primary_key=True)
    flow_global_id = peewee.CharField()
    source_node_id = peewee.CharField()
    destination_node_id = peewee.CharField()
    volume = peewee.BigIntegerField()
    arrival_time = peewee.IntegerField()

    class Meta:
        indexes = (
            (('flow_global_id', 'source_node_id', 'destination_node_id', 'volume', 'arrival_time'), True),
            # Note the trailing comma!
        )


class FlowCompletionTime(BaseModel):
    fct_id = peewee.BigAutoField(primary_key=True)
    fk_simulation = peewee.ForeignKeyField(Simulation, backref="flow_completion_times", on_delete="cascade")
    fk_flow = peewee.ForeignKeyField(Flow, backref="flow_completion_times")
    completion_time = peewee.DecimalField(max_digits=18, decimal_places=6)
    first_time_with_rate = peewee.DecimalField(max_digits=18, decimal_places=6)

    class Meta:
        indexes = (
            (('fk_simulation', 'fk_flow'), True),  # Note the trailing comma!
        )


TABLES = [ArrivalTimeGenerator, FlowVolumeGenerator, ConnectionPairGenerator, FlowGenerator, Topology, Algorithm,
          Simulation, Flow, FlowCompletionTime]
