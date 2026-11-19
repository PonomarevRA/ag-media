"""Compare local pronunciation and acoustic timing candidates on reviewed source cues."""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import time
import subprocess
import wave

from mlflow import MlflowClient
from voice_paths import temporary_directory
from voice_studio.speech.media_transcript import audio_offset
from voice_studio.synthesis.commercial_voice import REVISIONS, generate
from voice_studio.synthesis.dub_voice_profiles import isolate_reference_audio
from voice_studio.synthesis.engines import normalize
from voice_studio.synthesis.prosody_alignment import (RATE, align_phrases, fit_phrase, measured_contour, contour_error,
    read_pcm, reference_rhythm, transfer_intonation, write_pcm)
from voice_studio.synthesis.speech_metrics import measure
from voice_studio.synthesis.prosody_evaluation import pitch_promotion
from voice_studio.synthesis.video_dub import clean_cues, decode_original, source_streams


def digest(path):
    """Hash exact local inputs without storing audio in MLflow parameters."""
    import hashlib
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cut_cue(source, cue, offset, output):
    """Cut source-clock samples from the decoded audio packet clock."""
    with wave.open(str(source)) as audio:
        start=(cue['startMs']-offset)*24;stop=(cue['endMs']-offset)*24
        if not 0 <= start < stop <= audio.getnframes():
            raise ValueError('Cue is outside available PCM.')
        audio.setpos(start);write_pcm(output,audio.readframes(stop-start))


def create_tracker(output):
    """Create a separate durable local experiment without altering studio history."""
    client=MlflowClient(tracking_uri='sqlite:///'+str(output/'mlflow.db'))
    name='dubbing-prosody-'+output.name
    experiment=client.get_experiment_by_name(name)
    experiment_id=experiment.experiment_id if experiment else client.create_experiment(
        name,artifact_location=(output/'mlflow-artifacts').as_uri())
    return client,experiment_id


def metrics_for(source, candidate, text, root):
    """Keep ASR/identity and duration measurements separate from unknown accent/lip scores."""
    values=measure(root,'en',text,source,candidate)
    src,other=read_pcm(source),read_pcm(candidate)
    values['duration_error_ms']=abs(len(src)-len(other))*500/RATE
    plan=reference_rhythm(source)
    # Compare generated signal inside the exact measured quiet intervals, not just the clip length.
    energy=0;count=0
    from array import array
    for gap in plan['pauses']:
        raw=other[gap['startSample']*2:gap['endSample']*2]
        samples=array('h',raw)
        energy+=sum(x*x for x in samples);count+=len(samples)
    if count:values['source_pause_rms']= (energy/count)**.5
    return values


def generate_take(engine, mode, text, reference, seed, output):
    """Run one bounded offline generation and normalize the resulting WAV for analysis."""
    raw=output.with_name(output.stem+'-raw.wav');start=time.monotonic()
    job={'references':[reference],'output':str(raw),'text':text,'language':'en','seed':seed,
         'pronunciationMode':mode,'maxTokens':400}
    generate(engine,json.dumps(job),180)
    normalize(raw,output)
    return time.monotonic()-start


def write_browser_report(output, report):
    """Write a local A/B listening page with honest metric labels and original-frame references."""
    rows=[]
    for row in report['runs']:
        title=f"{row['engine']} · {row['stage']} · seed{row['seed']}"
        if row.get('promotion')=='rollback':title+=' · ROLLBACK'
        if row['status']!='ok':
            rows.append('<article><h3>'+escape(title)+'</h3><p>'+escape(row['error'])+'</p></article>');continue
        rows.append('<article><h3>'+escape(title)+'</h3><audio controls preload="metadata" src="'+escape(row['file'],quote=True)+'"></audio><pre>'+escape(json.dumps(row['metrics'],ensure_ascii=False,indent=2))+'</pre></article>')
    html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Просодия · A/B</title>
<style>body{font:16px/1.5 system-ui;background:#161b22;color:#e6edf3;max-width:1100px;margin:auto;padding:24px}h1{font-size:26px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px}article{border-top:2px solid #66c9bd;padding:12px 0;min-width:0}audio{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#66c9bd}</style>
<h1>Просодия · сравнительный пилот</h1><p>Один источник, одинаковые seed. «Нейтральный» — режим-кандидат; отсутствие акцента не подтверждено. Совпадение длительности и пауз не доказывает синхронизацию губ.</p><p>Сначала прослушайте оригинал, затем варианты. Для смены варианта остановите предыдущий проигрыватель.</p><audio controls src="reference.wav"></audio><p><a href="report.json">Технический отчёт и MLflow run IDs</a></p><main>'''+''.join(rows)+'</main>'
    (output/'index.html').write_text(html,encoding='utf-8')


def run(args):
    """Generate baseline/neutral pairs, then independently regenerate phrase-aligned alternatives."""
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    metadata=json.loads(args.report.read_text());duration,*_=source_streams(args.source)
    cues,_=clean_cues(metadata,duration);cue=cues[args.cue]
    reference_cue=cues[args.reference_cue if args.reference_cue is not None else args.cue]
    if cue['speaker']!=reference_cue['speaker']:
        raise ValueError('Reference cue belongs to a different speaker.')
    output.joinpath('input-cue.json').write_text(json.dumps(cue,ensure_ascii=False,indent=2))
    with temporary_directory(prefix='dubbing-pilot-') as temp:
        folder=Path(temp);original=folder/'source.wav';dialogue=folder/'dialogue.wav'
        decode_original(args.source,original,RATE);offset=audio_offset(args.source)
        isolate_reference_audio(original,args.background,dialogue,offset)
        cut_cue(dialogue,cue,offset,output/'reference.wav')
        cut_cue(dialogue,reference_cue,offset,output/'voice-reference.wav')
    reference={'path':str(output/'voice-reference.wav'),'text':reference_cue['sourceText']}
    plan=reference_rhythm(output/'reference.wav')
    client,experiment=create_tracker(output)
    report={'source':str(args.source.resolve()),'sourceSha256':digest(args.source),'cue':cue,
            'rhythm':plan,'referenceSha256':digest(output/'voice-reference.wav'),
            'models':REVISIONS,'runs':[],'accentVerdict':'human listening required',
            'lipSyncVerdict':'acoustic placement only; phonetic/visual synchrony unverified'}
    phrases=args.phrase or [cue['englishText']]
    for seed in args.seed:
        for engine in args.engine:
            for stage in ['source','neutral','phrases','intonation']:
                run=client.create_run(experiment,tags={'mlflow.runName':f'{engine}-{seed}-{stage}',
                    'source_sha256':report['sourceSha256'],'pair':str(args.cue)+'-'+str(seed),'stage':stage})
                run_id=run.info.run_id
                row={'engine':engine,'seed':seed,'stage':stage,'runId':run_id}
                path=output/f'{engine}-{seed}-{stage}.wav';start=time.monotonic()
                try:
                    if stage in ['source','neutral']:
                        row['generationSeconds']=generate_take(engine,stage,cue['englishText'],reference,seed,path)
                    elif stage=='phrases':
                        if len(phrases)!=len(plan['phrases']):
                            raise ValueError('Reviewed phrase translations must match the measured source islands.')
                        paths=[];row['phraseAttempts']=[]
                        for i,text in enumerate(phrases):
                            slot=plan['phrases'][i];accepted=None
                            for attempt in range(3):
                                piece=output/f'{engine}-{seed}-phrase-{i}-attempt-{attempt}.wav'
                                attempt_seed=seed+i+attempt*1009
                                generate_take(engine,'neutral',text,reference,attempt_seed,piece)
                                record={'phrase':i,'attempt':attempt,'seed':attempt_seed,'file':piece.name}
                                try:
                                    with temporary_directory(prefix='phrase-check-') as temp:
                                        fit_phrase(read_pcm(piece),slot['endSample']-slot['startSample'],Path(temp),.8,1.8)
                                    accepted=piece;record['status']='accepted'
                                except ValueError as error:
                                    record.update(status='rejected',reason=str(error))
                                row['phraseAttempts'].append(record)
                                if accepted:break
                            if accepted is None:raise ValueError('No phrase candidate fits safely after3 attempts.')
                            paths.append(accepted)
                        row['alignment']=align_phrases(paths,plan,path)
                    else:
                        baseline=next((x for x in reversed(report['runs']) if x['engine']==engine and x['seed']==seed and x['stage']=='phrases' and x['status']=='ok'),None)
                        if baseline is None:raise ValueError('No accepted phrase alignment to refine.')
                        previous=output/baseline['file']
                        row['pitch']=transfer_intonation(output/'reference.wav',previous,path,rhythm=plan)
                        if row['pitch']['status']!='applied':raise ValueError(row['pitch']['reason'])
                    row.update(status='ok',file=path.name,sha256=digest(path))
                    expected=' '.join(phrases) if stage in ['phrases','intonation'] else cue['englishText']
                    row['text']=expected
                    row['metrics']=metrics_for(output/'reference.wav',path,expected,Path.cwd())
                    if stage=='intonation':
                        for label,key in [('before','before'),('after','after')]:
                            error=contour_error(row['pitch']['source'],row['pitch'][key])
                            if error is not None:row['metrics']['f0_shape_'+label+'_semitones']=error
                        row['promotionAudit']=pitch_promotion(baseline['metrics'],row['metrics'])
                        row['promotion']=row['promotionAudit']['decision']
                        if row['promotion']=='rollback':row['rollbackFile']=baseline['file']
                        client.set_tag(run_id,'promotion',row['promotion'])
                    row['metrics']['wall_seconds']=time.monotonic()-start
                    client.log_batch(run_id,params=[],metrics=[],tags=[])
                    for key,value in {'engine':engine,'stage':stage,'seed':seed,'cue':args.cue}.items():client.log_param(run_id,key,value)
                    for key,value in row['metrics'].items():client.log_metric(run_id,key,float(value))
                    client.set_terminated(run_id)
                except (ValueError,OSError,subprocess.SubprocessError) as error:
                    row.update(status='rejected',error=str(error));client.set_tag(run_id,'rejection',str(error)[:500]);client.set_terminated(run_id,status='FAILED')
                report['runs'].append(row)
                artifact={key:value for key,value in row.items() if key!='text'}
                if 'text' in row:artifact['textSha256']=sha256(row['text'].encode()).hexdigest()
                client.log_dict(run_id,artifact,'result.json')
                (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                print(json.dumps(row,ensure_ascii=False),flush=True)
    write_browser_report(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','report','background','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cue',type=int,required=True);parser.add_argument('--reference-cue',type=int)
    parser.add_argument('--engine',action='append',choices=['qwen','chatterbox'],required=True)
    parser.add_argument('--seed',action='append',type=int,required=True)
    parser.add_argument('--phrase',action='append')
    run(parser.parse_args())
