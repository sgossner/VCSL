"""Create SFZ instruments from samples.

Written by Peter Eastman.  This file is in the public domain.

Usage: python createInstruments.py <instrument dir>

where <instrument dir> is the path to the top level directory containing all of the instrument's sample.  Output files
will be placed into the parent directory of <instrument dir>.
"""

from __future__ import print_function
from __future__ import division
import collections
import os
import sys
import numpy as np
import scipy.signal
import wavio

instrumentDir = sys.argv[1]
outputDir = os.path.abspath(os.path.join(instrumentDir, os.pardir))

noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
dynamicNames = ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'v11', 'vl2', 'vl3', 'vl4', 'vl5']

# Create objects to represent all the notes in the scale.

Note = collections.namedtuple('Note', ['index', 'name', 'frequency'])
noteWithName = {}
for octave in range(10):
    for i, name in enumerate(noteNames):
        index = i+(octave+1)*12
        name = name+str(octave)
        frequency = 440.0 * 2**((index-69)/12.0)
        noteWithName[name] = Note(index, name, frequency)

# Define a class to represent a single sample file.

class Sample(object):
    def __init__(self, filename):
        self.filename = filename
        self.note = None
        self.layer = None
        self.rr = None
        self.offset = 0
        self.tuning = 0
        self.loudness = 0
        fields = filename[:filename.index('.')].split('_')
        for field in fields:
            if field in noteWithName:
                self.note = noteWithName[field]
            elif field in dynamicNames:
                self.layer = dynamicNames.index(field)
            elif field.startswith('rr'):
                self.rr = int(field[2:])
            elif field.startswith('vl'):
                self.layer = int(field[2:])
            elif field[-1].isdigit() and field[:-1] in dynamicNames:
                self.layer = dynamicNames.index[field[:-1]]
                self.rr = int(field[-1])

# Define a class to represent an articulation.

class Articulation(object):
    def __init__(self, instrument, directory):
        self.instrument = instrument
        self.directory = directory
        self.samples = []
        self.noteSamples = collections.defaultdict(list)
        relpath = os.path.relpath(directory, instrument.directory)
        dirs = [d for d in os.path.split(relpath) if len(d) > 0]
        self.name = '-'.join(dirs)

    def addSample(self, sample):
        self.samples.append(sample);
        self.noteSamples[sample.note].append(sample)

# Define a class to represent an instrument.

class Instrument(object):
    def __init__(self, directory):
        self.directory = directory
        self.articulations = []
        head, tail = os.path.split(directory)
        if len(tail) > 0:
            self.name = tail
        else:
            self.name = os.path.split(head)[-1]

# The following functions are adapted from https://gist.github.com/endolith/255291.  They
# are used to estimate the pitch of a sample.

def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_autocorr(sig, fs):
    corr = scipy.signal.fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs / px

# Scan the directory to create a list of articulations and samples for the instrument.

def scanDirectory(directory, instrument):
    articulation = Articulation(instrument, directory)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            scanDirectory(filepath, instrument)
        elif filename.endswith('wav'):
            articulation.addSample(Sample(filename))
    if len(articulation.samples) > 0:
        instrument.articulations.append(articulation)

instrument = Instrument(instrumentDir)
scanDirectory(instrumentDir, instrument)

# Loop over all samples and analyze them.

for articulation in instrument.articulations:
    for sample in articulation.samples:
        wav = wavio.read(os.path.join(articulation.directory, sample.filename))
        
        # Identify silence at the start of the sample.
        
        amplitude = np.abs(np.max(wav.data, axis=1))
        smoothedAmplitude = np.convolve(amplitude, np.ones((100,))/100, mode='valid')
        cutoff = np.max(smoothedAmplitude)/50
        offset = np.min(np.where(smoothedAmplitude > cutoff))
        if offset > 100:
            sample.offset = offset-100
        
        # Identify the tuning correction.

        frequency = freq_from_autocorr(wav.data[:,0], wav.rate)
        for mult in [3, 2, 1, 0.5, 1/3.0]:
            f = frequency*mult
            ratio = f/sample.note.frequency
            if 0.95 < ratio < 1.05:
                sample.tuning = int(-100*np.log(ratio)/np.log(2**(1/12.0)))

        # Estimate the loudness as the 90th percentile of amplitude during the first second.

        sample.loudness = np.percentile(smoothedAmplitude[offset:offset+wav.rate], 90)

# Write an articulation to a SFZ file.

def writeArticulation(articulation, outfile):
    print('<group>', file=outfile)
    print('ampeg_attack=%f' % 0.1, file=outfile)
    print('ampeg_release=%f' % 0.5, file=outfile)
    print(file=outfile)
    
    # Find the range of keys to use each note for.

    notes = sorted(set([s.note for s in articulation.samples]))
    lowkey = np.zeros(len(notes))
    highkey = np.zeros(len(notes))
    lowkey[0] = notes[0].index
    for i in range(len(notes)-1):
        highkey[i] = (notes[i].index+notes[i+1].index)//2
        lowkey[i+1] = highkey[i]+1
    highkey[-1] = notes[-1].index+1

    # Compute a target loudness for every note.
    
    averageLoudness = np.zeros(len(notes))
    for noteIndex, note in enumerate(notes):
        samples = [s for s in articulation.samples if s.note == note]
        averageLoudness[noteIndex] = np.mean([s.loudness for s in samples])
    targetLoudness = np.zeros(len(notes))
    for i in range(len(notes)):
        lower = np.max([0, i-2])
        targetLoudness[i] = np.mean(averageLoudness[lower:i+3])

    # Loop over notes.

    for noteIndex, note in enumerate(notes):
        samples = [s for s in articulation.samples if s.note == note]
        layers = sorted(set([s.layer for s in samples]))
        
        # Loop over velocity layers for each note.
        
        for layerIndex, layer in enumerate(layers):
            lowvel = int(layerIndex*128/len(layers))
            highvel = int((layerIndex+1)*128/len(layers)-1)
            rrs = sorted(set([s.rr for s in samples if s.layer == layer]))
            
            # Loop over round robins for each layer.
            
            for rrIndex, rr in enumerate(rrs):
                sample = [s for s in samples if s.layer == layer and s.rr == rr]
                if len(sample) != 1:
                    raise ValueError('Found duplicate samples:', [s.filename for s in sample])
                sample = sample[0]
                print('<region>', file=outfile)
                if len(rrs) > 1:
                    print('seq_length=%d' % len(rrs), file=outfile)
                    print('seq_position=%d' % (rrIndex+1), file=outfile)
                if len(layers) > 1:
                    print('lovel=%d' % lowvel, file=outfile)
                    print('hivel=%d' % highvel, file=outfile)
                print('sample=%s' % os.path.relpath(os.path.join(articulation.directory, sample.filename), outputDir), file=outfile)
                print('pitch_keycenter=%s' % note.index, file=outfile)
                print('lokey=%s' % lowkey[noteIndex], file=outfile)
                print('hikey=%s' % highkey[noteIndex], file=outfile)
                if sample.offset > 0:
                    print('offset=%d' % sample.offset, file=outfile)
                if sample.tuning != 0:
                    print('tune=%d' % sample.tuning, file=outfile)
                amplification = targetLoudness[noteIndex]/sample.loudness
                db = 20*np.log10(amplification)
                if db > 6:
                    db = 6
                print('volume=%f' % db, file=outfile)
                print(file=outfile)
    
#  Create SFZ files for all the articulations.

for articulation in instrument.articulations:
    if len(instrument.articulations) == 1:
        filename = '%s.sfz' % instrument.name
    else:
        filename = '%s - %s.sfz' % (instrument.name, articulation.name)
    with open(os.path.join(outputDir, filename), 'w') as outfile:
        writeArticulation(articulation, outfile)

