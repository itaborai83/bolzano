import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import util
import kneebow.rotor as kbr

MIN_JOB_DURATION        = 60 
WINDOW_SIZE_FACTOR      = 0.075
WINDOW_SIZE_MIN         = 12
ELBOW_FINDING_EXIT_QTY  = 12
GRAPH_ASPECT            = 0.2
GRAPH_CM2INCH           = 0.393701
GRAPH_WIDTH             = 30.0 * GRAPH_CM2INCH
GRAPH_HEIGHT            = 10.0 * GRAPH_CM2INCH
LOG2                    = math.log(2.0)

FETCH_SQL = """
    SELECT  (START_DATE || ' ' || START_TIME)::TIMESTAMP AS START
    ,       DURATION
    FROM    SAP_JOBS
    WHERE   NAME        =  ?
    AND     START_DATE  >= ?
    AND     END_DATE    <= ?
    AND     DURATION    >= ?
    AND     STATUS      = 'F'
    ORDER   BY START_DATE
    ,       START_TIME
"""

FETCH_SQL_LIKE = """
    SELECT  (START_DATE || ' ' || START_TIME)::TIMESTAMP AS START
    ,       DURATION
    FROM    SAP_JOBS
    WHERE   NAME        LIKE ?
    AND     START_DATE  >= ?
    AND     END_DATE    <= ?
    AND     DURATION    >= ?
    AND     STATUS      = 'F'
    ORDER   BY START_DATE
    ,       START_TIME
"""


def run(job):
    start_date = '2019-01-01'
    end_date = '2020-01-01'
    cutoff_duration = MIN_JOB_DURATION
    execs = fetch_executions(job, start_date, end_date, cutoff_duration)
    winsize = max(int(len(execs) * WINDOW_SIZE_FACTOR), WINDOW_SIZE_MIN)
    scored_execs = score(execs, winsize)
    elbow = find_elbow(scored_execs, winsize)
    knee = find_knee(scored_execs, winsize)
    fill_threshold(scored_execs, elbow, knee)
    #scored_execs.dropna(inplace=True)
    scored_execs = scored_execs[ [ 'start', 'duration', 'upper_threshold', 'lower_threshold', 'ref_duration', 'score' ] ]
    scored_execs.to_csv('teste.tsv', sep='\t', decimal=',')
    save_graph(job, scored_execs)

def fetch_executions(job, start_date, end_date, cutoff_duration):
    sql = FETCH_SQL_LIKE if '%' in job else FETCH_SQL
    conn = util.get_db_conn('BOLZANO')
    cursor = conn.cursor()
    args = job, start_date, end_date, cutoff_duration
    cursor.execute(sql, args)
    starts = []
    durations = []
    for row in cursor:
        starts.append(row[0])
        durations.append(row[1])
    df = pd.DataFrame({ 'start': starts, 'duration': durations })
    return df

def score(execs, winsize):
    f = lambda x: math.log((1 + x.duration) / (1 + x.ref_duration)) / LOG2
    means = execs[ 'duration' ].rolling(winsize).mean().mul(1/3)
    medians = execs[ 'duration' ].rolling(winsize).median().mul(2/3)
    execs[ 'ref_duration' ] = means + medians
    execs[ 'score' ] = execs.apply(f, axis=1)
    return execs

def find_elbow(scored_execs, winsize):
    scored_execs = scored_execs.dropna()
    ys = scored_execs[ 'score' ].to_numpy()
    ys.sort()
    xs = np.linspace(ys.min(), ys.max(), len(ys))
    data = np.vstack( (xs, ys) ).T
    r = kbr.Rotor()
    r.fit_rotate(data)
    elbow = ys[ r.get_elbow_index() ]
    new_scored_execs = scored_execs[ scored_execs.score >= elbow ]
    if len(new_scored_execs) >= math.log(winsize)/LOG2 and len(new_scored_execs) != len(scored_execs):
        return find_elbow(new_scored_execs, winsize)
    else:
        return elbow

def find_knee(scored_execs, winsize):
    scored_execs = scored_execs.dropna()
    ys = scored_execs[ 'score' ].to_numpy()
    ys.sort()
    ys = ys[::-1]
    xs = np.linspace(ys.min(), ys.max(), len(ys))
    data = np.vstack( (xs, ys) ).T
    r = kbr.Rotor()
    r.fit_rotate(data)
    knee = ys[ r.get_knee_index() ]
    new_scored_execs = scored_execs[ scored_execs.score < knee ]
    if len(new_scored_execs) >= math.log(winsize)/LOG2 and len(new_scored_execs) != len(scored_execs):
        return find_elbow(new_scored_execs, winsize)
    else:
        return knee


def fill_threshold(scored_execs, elbow, knee):
    f_upper = lambda x: 2**elbow*x.ref_duration
    f_lower = lambda x: 2**knee*x.ref_duration
    scored_execs[ 'upper_threshold' ] = scored_execs.apply(f_upper, axis=1)
    scored_execs[ 'lower_threshold' ] = scored_execs.apply(f_lower, axis=1)

def save_graph(job, scored_execs):
    plt.close('all')
    plt.figure(num=None, figsize=(GRAPH_WIDTH, GRAPH_HEIGHT))
    #plt.yscale('log')
    plt.plot_date(scored_execs[ 'start' ], scored_execs[ 'duration' ], '+', label='Job Duration')
    plt.plot_date(scored_execs[ 'start' ], scored_execs[ 'upper_threshold' ], '--r', label='Upper Threshold')
    plt.plot_date(scored_execs[ 'start' ], scored_execs[ 'lower_threshold' ], '--r', label='Lower Threshold')
    plt.plot_date(scored_execs[ 'start' ], scored_execs[ 'ref_duration' ], '--r', label='Reference Duration')
    plt.xlabel('Start')
    plt.ylabel('Time(s)')
    plt.title(job)
    plt.grid()
    #plt.legend()
    plt.savefig('teste.png', )
