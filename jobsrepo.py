import numpy as np
import pandas as pd
import math
import util
import kneebow.rotor as kbr

FETCH_SQL = """
    SELECT  (START_DATE || ' ' || START_TIME)::TIMESTAMP AS START
    ,       DURATION
    FROM    SAP_JOBS
    WHERE   NAME = 'PB_FI0580N05A'
    AND     START_DATE  >= ?
    AND     END_DATE    <= ?
    AND     DURATION    >= ?
    AND     STATUS      = 'F'
    ORDER   BY START_DATE
    ,       START_TIME
"""


def fetch_executions(job, start_date, end_date, cutoff_duration):
    conn = util.get_db_conn('BOLZANO')
    cursor = conn.cursor()
    args = start_date, end_date, cutoff_duration
    cursor.execute(FETCH_SQL, args)
    starts = []
    durations = []
    for row in cursor:
        starts.append(row[0])
        durations.append(row[1])
    df = pd.DataFrame({
        'start': starts,
        'duration': durations
    })
    return df

def score(execs, winsize):
    log2 = math.log(2)
    f = lambda x: math.log(x.duration / x.ref_duration) / log2
    execs[ 'ref_duration' ] = execs[ 'duration' ].rolling(winsize).mean()
    execs[ 'score' ] = execs.apply(f, axis=1)
    return execs

def find_threshold(scored_execs, theta=45.0):
    theta = math.radians(theta)
    scored_execs = scored_execs.dropna()
    ys = scored_execs[ 'score' ].to_numpy()
    ys.sort()
    xs = np.linspace(ys.min(), ys.max(), len(ys))
    data = np.vstack( (xs, ys) ).T
    r = kbr.Rotor()
    r.fit_rotate(data, theta=theta)
    idx = r.get_elbow_index()
    elbow = ys[ idx ]
    return elbow


