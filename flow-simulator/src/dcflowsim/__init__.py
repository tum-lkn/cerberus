from decimal import *
from dcflowsim.constants import DECIMAL_PRECISION

# Set the precision for the Decimal calculations context for the complete simulator
getcontext().prec = DECIMAL_PRECISION
# Set up the Decimal context to fail at creation from float or comparison with float
getcontext().traps[FloatOperation] = True

