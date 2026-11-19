"""Compare coarse viseme drafts; eSpeak G2P does not supply measured phone timestamps."""
import argparse
import ctypes
from importlib.metadata import version
import json
from pathlib import Path
import re


def phonemes(library,text,language):
    """Request separated IPA hypotheses from the locally installed eSpeak library."""
    if library.espeak_SetVoiceByName(language.encode()):raise ValueError('Unknown eSpeak voice.')
    cursor=ctypes.c_char_p(text.encode());result=[]
    while cursor.value:
        before=cursor.value
        raw=library.espeak_TextToPhonemes(ctypes.byref(cursor),1,2|(ord('|')<<8))
        for word in (raw or b'').decode().split():
            result.extend(x for x in re.sub('[ˈˌː]','',word).split('|') if x)
        if cursor.value==before:raise ValueError('G2P did not advance.')
    return result


def visemes(phones):
    """Project IPA phones to a deliberately coarse mouth-shape hypothesis, without timing claims."""
    groups=[('closed','pbm'),('lip-teeth','fv'),('rounded','uoɔʊwy'),('open','aɑæɐʌ'),('spread','iɪeɛɨ')]
    result=[]
    for phone in phones:
        group='neutral'
        for label,characters in groups:
            if any(c in characters for c in phone):group=label;break
        result.append(group)
    return result


def edit_cost(left,right):
    """Return normalized sequence edit cost; this is a draft heuristic, not a lip-sync score."""
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        row=[i]
        for j,b in enumerate(right,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=row
    return previous[-1]/max(1,len(left),len(right))


def main(args):
    """Persist original/variant IPA and costs for a human-reviewed translation comparison."""
    import espeakng_loader
    lib=ctypes.CDLL(espeakng_loader.get_library_path())
    lib.espeak_Initialize(2,0,str(Path(espeakng_loader.get_data_path()).parent).encode(),0)
    lib.espeak_TextToPhonemes.restype=ctypes.c_char_p
    lib.espeak_TextToPhonemes.argtypes=[ctypes.POINTER(ctypes.c_char_p),ctypes.c_int,ctypes.c_int]
    source=phonemes(lib,args.source,'ru');shape=visemes(source)
    rows=[]
    for text in args.english:
        phones=phonemes(lib,text,'en-us');draft=visemes(phones)
        rows.append({'text':text,'ipaPhones':phones,'visemeDraft':draft,'sequenceEditCost':edit_cost(shape,draft),'phoneCountRatio':len(phones)/max(1,len(source))})
    args.output.write_text(json.dumps({'g2p':'espeakng-loader'+version('espeakng-loader'),'sourceText':args.source,'sourcePhones':source,'sourceVisemes':shape,'variants':rows,'limitations':'G2P and coarse sequence heuristic only. No acoustic forced alignment, observed lips or perceptual verdict. Separate process pilot, not a bundled G2P dependency.'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--english',action='append',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
