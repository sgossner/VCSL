"""Create SFZ instruments from samples.

Written by Peter Eastman.  This file is in the public domain.  For usage instructions, run this script as

python createInstruments.py --help

This script use the PySoundFile library.  See https://pypi.org/project/PySoundFile/ for installation instructions.
You can still use it without PySoundFile, but only wav files will be supported.
"""

from __future__ import print_function
from __future__ import division
import argparse
import collections
import os
import sys
import numpy as np
import scipy.signal

try:
    # Use PySoundFile if possible.
    import soundfile as sf
    formats = ['.wav', '.ogg', '.flac']
    def loadFile(path):
        return sf.read(path)
except:
    # It isn't installed, so use wavio instead.
    import wavio
    formats = ['.wav']
    def loadFile(path):
        wav = wavio.read(path)
        return wav.data, wav.rate


noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
dynamicNames = ['pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff',
                'v11', 'vl2', 'vl3', 'vl4', 'vl5',
                'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9',
                'soft', 'quiet', 'med', 'medium', 'loud']

# Parse command line options.

parser = argparse.ArgumentParser(description='Create SFZ instruments from samples.')
parser.add_argument('--attack', type=float, default=0.004, help='Attack time in seconds (default=0.004)')
parser.add_argument('--release', type=float, default=0.3, help='Release time in seconds (default=0.3)')
parser.add_argument('--velocityexponent', type=float, default=0.6, help='Exponent shaping the curve assigning layers to velocities (default=0.6)')
parser.add_argument('--transpose', type=int, default=0, help='Transposition that has been applied to the samples, in semitones (default=0)')
parser.add_argument('--volume', type=float, default=0.0, help='Amplification to apply to all samples, in dB (default=0.0)')
parser.add_argument('--releasevolume', type=float, default=0.0, help='Amplification to apply to release samples, in dB (default=0.0)')
parser.add_argument('--crossfade', action='store_true', help='Perform crossfading between velocity layers')
parser.add_argument('--unpitched', action='store_true', help='Treat this as an unpitched instrument')
parser.add_argument('--noreleases', action='store_true', help='Ignore release samples')
parser.add_argument('--notuning', action='store_true', help='Do not apply tuning to correct pitches')
parser.add_argument('--articulation', type=str, default=None, help='Name of the articulation to create (default is to create all articulations)')
parser.add_argument('instrumentdir', help="Top level directory containing the instrument's samples")
args = parser.parse_args()
outputDir = os.path.abspath(os.path.join(args.instrumentdir, os.pardir))

# Create objects to represent all the notes in the scale.

Note = collections.namedtuple('Note', ['index', 'name', 'frequency'])
noteWithName = {}
for octave in range(-1, 10):
    for i, name in enumerate(noteNames):
        index = i+(octave+2)*12-args.transpose
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
        noteFields = []
        fields = filename[:filename.index('.')].split('_')
        for field in fields:
            if field in noteWithName and not args.unpitched:
                self.note = noteWithName[field]
            elif field in dynamicNames:
                self.layer = dynamicNames.index(field)
            elif field.startswith('rr'):
                self.rr = int(field[2:])
            elif field.startswith('vl'):
                self.layer = int(field[2:])
            elif field[-1].isdigit() and field[:-1] in dynamicNames:
                self.layer = dynamicNames.index(field[:-1])
                self.rr = int(field[-1])
            else:
                noteFields.append(field)
        if args.unpitched:
            # For unpitched instrument, two samples are considered the same "note" if all parts of the filename
            # match except layer and round robin.
            self.note = tuple(noteFields)

# Define a class to represent an articulation.

class Articulation(object):
    def __init__(self, instrument, directory, isRelease):
        self.instrument = instrument
        self.directory = directory
        self.isRelease = isRelease
        self.samples = []
        self.name = os.path.split(directory)[-1]
        if isRelease and self.name == 'Releases':
            # This happens when there's a single articulation, with the samples directly in the "Sustains" and "Releases" folders.
            self.name = 'Sustains'

    def addSample(self, sample):
        self.samples.append(sample);

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

# This function computes the low end of the velocity range for a layer.

def layer_lowvel(layerIndex, layers):
    numLayers = len(layers)
    if args.crossfade:
        numLayers -= 1
    return int((layerIndex/numLayers)**args.velocityexponent*128)

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

def scanDirectory(directory, instrument, isRelease):
    if os.path.split(directory)[-1].lower() == 'releases':
        if args.noreleases:
            return
        isRelease = True
    articulation = Articulation(instrument, directory, isRelease)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            scanDirectory(filepath, instrument, isRelease)
        elif any(filename.endswith(e) for e in formats):
            articulation.addSample(Sample(filename))
    if len(articulation.samples) > 0:
        if args.articulation is None or articulation.name == args.articulation:
            instrument.articulations.append(articulation)

instrument = Instrument(args.instrumentdir)
scanDirectory(args.instrumentdir, instrument, False)

# Loop over all samples and analyze them.

for articulation in instrument.articulations:
    for sample in articulation.samples:
        data, rate = loadFile(os.path.join(articulation.directory, sample.filename))
        
        # Identify silence at the start of the sample.
        
        amplitude = np.abs(np.max(data, axis=1))
        smoothedAmplitude = np.convolve(amplitude, np.ones((100,))/100, mode='valid')
        cutoff = np.max(smoothedAmplitude)/50
        offset = np.min(np.where(smoothedAmplitude > cutoff))
        if offset > 100:
            sample.offset = offset-100
        
        # Identify the tuning correction.

        if not args.unpitched and not args.notuning:
            frequency = freq_from_autocorr(data[:,0], rate)
            for mult in [3, 2, 1, 0.5, 1/3.0]:
                f = frequency*mult
                ratio = f/sample.note.frequency
                if 0.95 < ratio < 1.05:
                    sample.tuning = int(-100*np.log(ratio)/np.log(2**(1/12.0)))

        # Estimate the loudness as the 90th percentile of amplitude during the first second.

        sample.loudness = np.percentile(smoothedAmplitude[offset:offset+rate], 90)

# Write an articulation to a SFZ file.

def writeArticulation(articulation, outfile, isSustain, isRelease):
    print('<group>', file=outfile)
    attack = args.attack
    release = args.release
    if isSustain:
        print('trigger=attack', file=outfile)
        release = 0.1
    elif isRelease:
        print('trigger=release', file=outfile)
        attack = 0.1
    print('ampeg_attack=%f' % attack, file=outfile)
    if not isRelease:
        print('ampeg_release=%f' % release, file=outfile)
    print(file=outfile)
    
    # Find the range of keys to use each note for.

    notes = sorted(set([s.note for s in articulation.samples]))
    if args.unpitched:
        lowkey = np.arange(60, 60+len(notes))
        highkey = lowkey
    else:
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
    if args.unpitched:
        targetLoudness = averageLoudness
    else:
        targetLoudness = np.zeros(len(notes))
        for i in range(len(notes)):
            lower = np.max([0, i-2])
            targetLoudness[i] = np.mean(averageLoudness[lower:i+3])

    # Loop over notes.

    for noteIndex, note in enumerate(notes):
        samples = [s for s in articulation.samples if s.note == note]
        
        # Loop over velocity layers for each note.
        
        layers = sorted(set([s.layer for s in samples]))
        for layerIndex, layer in enumerate(layers):
            
            # Loop over round robins for each layer.
            
            rrs = [s for s in samples if s.layer == layer]
            for rrIndex, sample in enumerate(rrs):
                print('<region>', file=outfile)
                if len(rrs) > 1:
                    print('seq_length=%d' % len(rrs), file=outfile)
                    print('seq_position=%d' % (rrIndex+1), file=outfile)
                if len(layers) > 1:
                    if args.crossfade:
                        if layerIndex > 0:
                            print('xfin_lovel=%d' % layer_lowvel(layerIndex-1, layers), file=outfile)
                            print('xfin_hivel=%d' % (layer_lowvel(layerIndex, layers)-1), file=outfile)
                        if layerIndex < len(layers)-1:
                            print('xfout_lovel=%d' % layer_lowvel(layerIndex, layers), file=outfile)
                            print('xfout_hivel=%d' % (layer_lowvel(layerIndex+1, layers)-1), file=outfile)
                    else:
                        print('lovel=%d' % layer_lowvel(layerIndex, layers), file=outfile)
                        print('hivel=%d' % (layer_lowvel(layerIndex+1, layers)-1), file=outfile)
                print('sample=%s' % os.path.relpath(os.path.join(articulation.directory, sample.filename), outputDir), file=outfile)
                if args.unpitched:
                    keycenter = lowkey[noteIndex]
                else:
                    keycenter = note.index
                print('pitch_keycenter=%s' % keycenter, file=outfile)
                print('lokey=%d' % lowkey[noteIndex], file=outfile)
                print('hikey=%d' % highkey[noteIndex], file=outfile)
                if sample.offset > 0:
                    print('offset=%d' % sample.offset, file=outfile)
                if sample.tuning != 0 and not args.notuning:
                    print('tune=%d' % sample.tuning, file=outfile)
                if articulation.isRelease:
                    volume = args.releasevolume
                else:
                    volume = args.volume
                amplification = targetLoudness[noteIndex]/sample.loudness
                db = 20*np.log10(amplification) + volume
                print('volume=%f' % db, file=outfile)
                print(file=outfile)
    
#  Create SFZ files for all the articulations.

articulations = [a for a in instrument.articulations if not a.isRelease]
for articulation in articulations:
    if not articulation.isRelease:
        if len(articulations) > 1:
            filename = '%s - %s.sfz' % (instrument.name, articulation.name)
        elif args.articulation is not None:
            filename = '%s - %s.sfz' % (instrument.name, args.articulation)
        else:
            filename = '%s.sfz' % instrument.name
        
        # See if we have release samples for this articulation.
        
        release = None
        for a in instrument.articulations:
            if a.isRelease and a.name == articulation.name:
                release = a
        with open(os.path.join(outputDir, filename), 'w') as outfile:
            print('// Generation Options:', ' '.join(sys.argv[1:-1]), file=outfile)
            print(file=outfile)
            writeArticulation(articulation, outfile, release is not None, False)
            if release is not None:
                writeArticulation(release, outfile, False, True)
