import PySimpleGUI as sg
import os
from pygame import mixer
from time import sleep

def GUI_Builder():
    sg.theme('DarkBrown2')

    layout = []
    layoutcol = []

    # buttons
    for file in os.listdir('samples'):
        # buttons
        layout.append([sg.Button(file[:-4])])
        # checkboxes
        layoutcol.append([sg.Checkbox(file[:-4], default=True, key=file[:-4])])

    col1 = sg.Column(layout, element_justification='c', vertical_scroll_only=True,scrollable=True)
    col2 = sg.Column(layoutcol, element_justification='c', vertical_scroll_only=True,scrollable=True)
    col = [[col1, col2]]
    layout = [[sg.Text('Click the button to play a sound')]]
    layout.append(col)
    layout.append([sg.Button('Play Sound')])
    wind = sg.Window('Sound Player', layout= layout, grab_anywhere=True)
    return wind


def stringDict():
    # d = {0 : 'E4.wav', 1 : 'B3.wav', 2 : 'G3.wav', 3 : 'D3.wav', 4 : 'A2.wav', 5 : 'E2.wav'}
    # d = {'e' : 'E4.wav', 'B' : 'B3.wav', 'G' : 'G3.wav', 'D' : 'D3.wav', 'A' : 'A2.wav', 'E' : 'E2.wav'}
    d = {'E' : 'E2.wav', 'A' : 'A2.wav', 'D' : 'D3.wav', 'G' : 'G3.wav', 'B' : 'B3.wav', 'e' : 'E4.wav' }
    return d


def player(d):
    index = 0
    for key in d:
        mixer.Channel(index).play(mixer.Sound('samples/' + d[key]))
        index += 1
def Em():
    d = stringDict()
    d['A'] = "B2.wav"
    d['D'] = "E3.wav"
    player(d)

def D():
    d = stringDict()
    # removeing stringe
    e = d.pop('E')
    a = d.pop('A')
    d['G'] = 'A3.wav'
    d['B'] = 'D4.wav'
    d['e'] = 'Fs4.wav'
    player(d)
    # index = 0
    # for key in d:
    #     mixer.Channel(index).play(mixer.Sound('samples/' + d[key]))
    #     index += 1

def G():
    d = stringDict()
    d["E"] = "G2.wav"
    d["e"] = "G4.wav"
    d["A"] = "B2.wav"
    player(d)

def C():
    d = stringDict()
    d.pop('E')
    d["A"] = "C3.wav"
    d["D"] = "E3.wav"
    d["B"] = "C4.wav"
    player(d)

def main():
    window = GUI_Builder()
    samplelist = []
    for file in os.listdir('samples'):
        samplelist.append(file[:-4])
    print(samplelist)

    while True:
        event, values = window.read()
        print(event)
        if event == sg.WIN_CLOSED:
            break
        if event in samplelist:
            print('Playing sound', event)
            # give varible e2 the sound
            sound = mixer.Sound('samples/' + event + '.wav')
            sound = mixer.Channel(0).play(sound)
            # sound.play()
        if event == 'Play Sound':
            print('Playing sound')
            loop = 0
            while loop < 3:
                Em()
                sleep(0.2)
                Em()
                sleep(0.2)
                Em()
                sleep(0.2)
                Em()
                sleep(0.2)
                C()
                sleep(0.2)
                C()
                sleep(0.2)
                C()
                sleep(0.2)
                C()
                sleep(0.2)
                G()
                sleep(0.2)
                G()
                sleep(0.2)
                G()
                sleep(0.2)
                G()
                sleep(0.2)
                D()
                sleep(0.2)
                D()
                sleep(0.2)
                D()
                sleep(0.2)
                D()
                sleep(0.2)
                loop += 1
                if event == sg.WIN_CLOSED:
                    break


       
if __name__ == '__main__':
    mixer.init()
    main()


