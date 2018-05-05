"""Create keyswitch files by merging the SFZ files for individual articulations.

Written by Peter Eastman.  This file is in the public domain.
"""

import os
import re

# Read the content of a single SFZ file.

def readFile(directory, instrument, articulation):
    path = os.path.join(directory, '%s - %s.sfz' % (instrument, articulation))
    with open(path)as infile:
        return infile.read()

# This function recursively scans directories to create keyswitch files.

def scanDirectory(directory):
    files = os.listdir(directory)
    matches = []
    for f in files:
        # Recursively scan subdirectories.

        filePath = os.path.join(directory, f)
        if os.path.isdir(filePath):
            scanDirectory(filePath)

        # Find SFZ files for instruments with multiple articulations.

        match = re.match('(.*) - (.*)\.sfz', f)
        if match is not None:
            matches.append(match)

    # Identify instruments and the articulations for each one.

    instruments = set([m.group(1) for m in matches])
    for instrument in instruments:
        articulations = [m.group(2) for m in matches if m.group(1) == instrument]
        
        # Scan the files to remove any with keyswitches, and to find the range of notes.
        
        lowestKey = 127
        highestKey = 0
        for articulation in articulations[:]:
            content = readFile(directory, instrument, articulation)
            if 'sw_last' in content:
                articulations.remove(articulation)
                continue
            for m in re.findall('lokey=(.*)$', content, re.MULTILINE):
                lowestKey = min(lowestKey, int(m))
                highestKey = max(highestKey, int(m))
            for m in re.findall('hikey=(.*)$', content, re.MULTILINE):
                lowestKey = min(lowestKey, int(m))
                highestKey = max(highestKey, int(m))
        if len(articulations) < 2:
            continue

        # Decide where on the keyboard to put the keyswitches.
        
        if lowestKey >= 51+len(articulations):
            baseKey = 48
        elif lowestKey >= 39+len(articulations):
            baseKey = 36
        elif highestKey < 69:
            baseKey = 72
        elif highestKey < 81:
            baseKey = 84
        elif highestKey < 97-len(articulations):
            baseKey = 97-len(articulations)
        else:
            bestLow = lowestKey-len(articulations)-3
            bestHigh = highestKey+3
            if abs(67-bestLow) < abs(bestHigh-67):
                baseKey = bestLow
            else:
                baseKey = bestHigh

        # Try to come up with a reasonable order for the articulations.  Look for names like "normal", "sustain", or
        # "staccato", then just put shorter ones before longer ones.

        groups = [[] for i in range(5)]
        for articulation in articulations:
            name = articulation.lower()
            if 'norm' in name:
                groups[0].append(articulation)
            elif 'stac' in name:
                groups[3].append(articulation)
            elif 'vib' in name:
                groups[2].append(articulation)
            elif 'sus' in name:
                groups[1].append(articulation)
            else:
                groups[4].append(articulation)
        articulations = sum((sorted(g, key=lambda x: (len(x), x)) for g in groups), [])

        # Generate the keyswitch file by merging the individual articulations.
        
        path = os.path.join(directory, '%s - Keyswitch.sfz' % instrument)
        with open(path, 'w') as output:
            for i, articulation in enumerate(articulations):
                content = readFile(directory, instrument, articulation)
                commands = """
sw_default=%d
sw_lokey=%d
sw_hikey=%d
sw_last=%d
sw_label=%s""" % (baseKey, baseKey, baseKey+len(articulations), baseKey+i, articulation)
                content = content.replace('<group>', '<group>'+commands)
                output.write(content)
                output.write('\n\n')

# Generate files, starting from the top of the repository.

rootDir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
scanDirectory(rootDir)
