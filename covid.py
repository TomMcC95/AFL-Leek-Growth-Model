from requests import get
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from matplotlib.widgets import Cursor
import matplotlib.ticker as ticker

def get_data(url):
    response = get(endpoint, timeout=10)
    
    if response.status_code >= 400:
        raise RuntimeError(f'Request failed: { response.text }')
        
    return response.json()


class SnaptoCursor(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3) 
        self.x = x
        self.y = y
        self.txt = ax.text(0, 0, '')

    def mouse_move(self, event):
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()
    

if __name__ == '__main__':
    endpoint = (
        'https://api.coronavirus.data.gov.uk/v1/data?'
        'filters=areaType=nation;areaName=england&'
        'structure={"date":"date","newCases":"newCasesByPublishDate", "newTests":"newPCRTestsByPublishDate", "newHospital":"hospitalCases", "newDeaths":"newDeaths28DaysByDeathDate"}'
    )
    
    
    df = pd.json_normalize(get_data(endpoint), record_path =['data'])
    df = df[::-1].reset_index(drop = True)
    df['new100Cases'] = df['newCases']/100
    df['new100Hospital'] = df['newHospital']/100
    df['new1000Tests'] = df['newTests']/1000
    print(df.tail())

    
    
    df['new100Cases'].plot(label='new100Cases', color='orange')
    df['new1000Tests'].plot(label='new1000Tests')
    df['new100Hospital'].plot(label='new100Hospital')
    df['newDeaths'].plot(label='newDeaths')
    
    plt.title('Covid Data')
    plt.xlabel('Day')
    plt.legend()
    plt.show()

    rolling_windows = df['new100Cases'].rolling(7)
    df['new100Cases_rolling'] = rolling_windows.mean()
    rolling_windows = df['newDeaths'].rolling(7)
    df['newDeaths_rolling'] = rolling_windows.mean()
    rolling_windows = df['new100Hospital'].rolling(7)
    df['new100Hospital_rolling'] = rolling_windows.mean()
    rolling_windows = df['new1000Tests'].rolling(7)
    df['new1000Tests_rolling'] = rolling_windows.mean()



    df['new100Cases_rolling'].plot(label='new100Cases_rolling', color='orange')
    df['new1000Tests_rolling'].plot(label='new1000Tests_rolling')
    df['new100Hospital_rolling'].plot(label='new100Hospital_rolling')
    df['newDeaths_rolling'].plot(label='newDeaths_rolling')

    tick_array = [i*25 for i in range((len(df['new100Cases'])//25)+2)]

    plt.xticks(tick_array)
    plt.grid(True, markevery = 50)
    plt.xlim(0)
    plt.ylim(0)
    plt.title('Covid Data Rolling')
    plt.xlabel('Day')
    plt.legend()
    plt.show()


    fig, ax = plt.subplots()

    cursor = SnaptoCursor(ax, df['date'].index, df['newDeaths'])
    cid =  plt.connect('motion_notify_event', cursor.mouse_move)

    ax.plot(df['newDeaths'])
    plt.show()

    