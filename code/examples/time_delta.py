from datetime import datetime
import humanfriendly
from datetime import timedelta
delta = timedelta(seconds = 321)
humanfriendly.format_timespan(delta)

print(delta)
start_time = datetime.now()

time_elapsed = datetime.now() - start_time
print(time_elapsed)