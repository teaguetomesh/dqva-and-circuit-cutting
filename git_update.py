# import argparse
import subprocess
from apscheduler.schedulers.blocking import BlockingScheduler

# parser = argparse.ArgumentParser(description='git updates')
# parser.add_argument('--comments', type=str, default='updates', metavar='S',
# help='git commit comments (default: updates)')
# args = parser.parse_args()

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=45)
def timed_job():
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'commit', '-m', '"Regular updates"'])
    subprocess.call(['git', 'push'])
    print('Regular push every 45 minutes.')

@sched.scheduled_job('cron', day_of_week='mon-sun', hour=17, minute=15)
def scheduled_job():
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'commit', '-m', '"Daily final push"'])
    subprocess.call(['git', 'push'])
    print('Daily final push at 5:15pm.')

# sched.configure(options_from_ini_file)
sched.start()