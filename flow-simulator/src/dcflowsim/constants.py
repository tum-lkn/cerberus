# Precision for floating point rounding to Decimal
FLOAT_DECIMALS = 6
# Overall maximum significant digits for Decimal calculations in the simulator
DECIMAL_PRECISION = 18

# RotorSwitch Reconfiguration Event Period
ROTORSWITCH_RECONF_PERIOD = 1
ROTORSWITCH_NIGHT_DURATION = 0

INTERFACE_TYPE_DATABASE = "peewee"
PEEWEE_INTERFACE_INSERT_MANY_SIZE = 1000  # Number of elements per insert many statement

# Priorities of environment listeners
STATISTICS_COLLECTOR_PRIORITY = 30
EVENT_INSERTION_PRIORITY = 25
ALGORITHM_PRIORITY = 20
ENVIRONMENT_VIEW_PRIORITY = 10

MAX_NUM_EVENTS = 1e5

SIMULATION_STATUS_MESSAGE_FREQUENCY = 100

LOGGING_PATH = "logging.yaml"