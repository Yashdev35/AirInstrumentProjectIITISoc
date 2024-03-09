'''
Chord-Maker.py

This is a simple program to test out the sound files in the samples folder.
They have the sound of a guitar string being plucked.

# Aim:
The Goal is to make a program that can play the sound of a guitar chords by choosing the right combination of sound files.

# Current State:
The program can play the sound of the guitar strings by clicking the buttons.
The program can also play a song using the chords made from the sound files.
Chords present are
    - Em
    - D
    - G
    - C
'''



import PySimpleGUI as sg
import os
from pygame import mixer
from time import sleep
from pygame import mixer


def GUI_Builder() -> sg.Window:
    '''
    Description:
    ------------
    This function builds the GUI for the sound player application.

    Returns:
    --------
    sg.Window: The window object representing the GUI.
    '''
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

# Mixer is used in this functions as it allows multiple sounds to be played at the same time.
def stringDict() -> dict:
    '''
    Description:
    ------------
    This function returns a dictionary with the string notes mapped to the sound files as values.
    The keys are the string notes and the values are the sound files.

    Returns:
    --------
    dict: A dictionary with the string notes as keys and the sound files as values.
    '''
    # d = {0 : 'E4.wav', 1 : 'B3.wav', 2 : 'G3.wav', 3 : 'D3.wav', 4 : 'A2.wav', 5 : 'E2.wav'}
    # d = {'e' : 'E4.wav', 'B' : 'B3.wav', 'G' : 'G3.wav', 'D' : 'D3.wav', 'A' : 'A2.wav', 'E' : 'E2.wav'}
    d = {'E' : 'E2.wav', 'A' : 'A2.wav', 'D' : 'D3.wav', 'G' : 'G3.wav', 'B' : 'B3.wav', 'e' : 'E4.wav' }
    return d

def player(d) -> None:
    """
    Description:
    ------------
    This function plays the sound files mapped in the dictionary d.

    Parameters:
    -----------
    d : dict
        A dictionary with the string notes as keys and the sound files as values.

    Returns:
    --------
    None
    """
    index = 0
    for key in d:
        mixer.Channel(index).play(mixer.Sound('samples/' + d[key]))
        index += 1
def Em():
    '''
    Description:
    ------------
    This function plays the sound of the Em chord.
    '''
    d = stringDict()
    d['A'] = "B2.wav"
    d['D'] = "E3.wav"
    player(d)

def D():
    '''
    Description:
    ------------
    This function plays the sound of the D chord.
    '''
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
    '''
    Description:
    ------------
    This function plays the sound of the G chord.
    '''
    d = stringDict()
    d["E"] = "G2.wav"
    d["e"] = "G4.wav"
    d["A"] = "B2.wav"
    player(d)

def C():
    '''
    Description:
    ------------
    This function plays the sound of the C chord.
    '''
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
        event, _ = window.read()
        print(event)

        # if the window is closed, break the loop
        if event == sg.WIN_CLOSED:
            break
        
        # play the sound of the button clicked
        if event in samplelist:
            print('Playing sound', event)
            # give varible e2 the sound
            sound = mixer.Sound('samples/' + event + '.wav')
            sound = mixer.Channel(0).play(sound)
            # sound.play()

        # play a song using chords made from the sound files
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


