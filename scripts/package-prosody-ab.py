"""Package a pilot as timed A/B videos while preserving the source background bed."""
import argparse
from pathlib import Path
import json
import shutil

from voice_paths import temporary_directory
from voice_studio.synthesis.dub_background import mix_english
from voice_studio.synthesis.engines import ffmpeg,normalize
from voice_studio.synthesis.prosody_alignment import RATE,align_phrases


def write_ab_page(output, rows):
    """Offer source-clock A/B switching while retaining the paused/playing position."""
    from html import escape
    buttons=[]
    for index,row in enumerate(rows):
        buttons.append('<button type="button" data-video="'+escape(row['video'],quote=True)+
                       '" aria-pressed="'+str(index==0).lower()+'">'+escape(row['label'])+'</button>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Синтез · A/B</title>
<style>body{font:16px/1.5 system-ui;background:#161b22;color:#e6edf3;max-width:1000px;margin:auto;padding:20px}video{display:block;max-height:65vh;width:100%;background:#000}nav{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}button{font:inherit;padding:8px 12px;color:inherit;background:#28323e;border:1px solid #667a8b;border-radius:5px}button[aria-pressed=true]{color:#86e0d3;border-color:currentColor}button:focus-visible{outline:2px solid #86e0d3;outline-offset:3px}</style>
<h1>Синтез · A/B</h1><p>Переключение сохраняет позицию. Нейтральный режим — кандидат, не подтверждённый результат без акцента. F0 с пометкой rolled back исключён из автоматического выбора.</p>
<video controls preload="metadata" src="original.mp4"></video><nav aria-label="Варианты">__BUTTONS__</nav>
<button id="ab-back" type="button">−1 с</button> <button id="ab-forward" type="button">+1 с</button><label>Общая позиция<input id="ab-position" type="range" min="0" max="0" step="0.1" value="0" style="display:block;width:100%"></label>
<p>Сравнивайте произношение, паузы и движения губ отдельно: точная длина WAV не доказывает фонетическую синхронизацию.</p>
<script>
const video=document.querySelector('video'),range=document.querySelector('#ab-position');
video.addEventListener('loadedmetadata',()=>{range.max=video.duration;range.value=video.currentTime;});
video.addEventListener('timeupdate',()=>{range.value=video.currentTime;});
let pending=null,generation=0;
range.oninput=()=>{video.currentTime=Number(range.value);};
document.querySelector('#ab-back').onclick=()=>{video.currentTime=Math.max(0,video.currentTime-1);};
document.querySelector('#ab-forward').onclick=()=>{video.currentTime=Math.min(video.duration,video.currentTime+1);};
for(const button of document.querySelectorAll('button[data-video]')) button.onclick=()=>{
    const state=pending??{position:video.currentTime,playing:!video.paused};
    pending=state;const current=++generation;
    video.pause();video.src=button.dataset.video;
    for(const other of document.querySelectorAll('button[data-video]')) other.setAttribute('aria-pressed',String(other===button));
    video.addEventListener('loadedmetadata',()=>{
        if(current!==generation)return;
        video.currentTime=Math.min(state.position,video.duration);pending=null;
        if(state.playing)video.play().catch(()=>{});
    },{once:true});
};
</script>'''
    (output/'index.html').write_text(page.replace('__BUTTONS__',''.join(buttons)),encoding='utf-8')


def package(args):
    """Render labeled source/conditioning/timing comparisons without changing previous exports."""
    folder=args.pilot.resolve();report=json.loads((folder/'report.json').read_text());cue=report['cue']
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    rows=[x for x in report['runs'] if x['engine']==args.engine and x['seed']==args.seed and x['status']=='ok']
    by_stage={x['stage']:x for x in rows}
    duration=cue['endMs']-cue['startMs'];start=cue['startMs']/1000
    metadata=[]
    with temporary_directory(prefix='prosody-ab-') as name:
        temp=Path(name);original=temp/'original.wav';bed=temp/'bed.wav'
        ffmpeg(['-ss',str(start),'-i',report['source'],'-t',str(duration/1000),'-vn','-ac','2','-ar','44100','-c:a','pcm_s24le',str(original)])
        ffmpeg(['-ss',str(start),'-i',str(args.background.resolve()),'-t',str(duration/1000),'-c:a','pcm_s24le',str(bed)])
        variants=[('original','Original',original)]
        for stage,label in [('source','Reference accent'),('neutral','Neutral candidate'),('phrases','Phrase timing'),('intonation','F0 candidate')]:
            if stage not in by_stage:continue
            row=by_stage[stage];path=folder/row['file']
            if stage=='intonation' and row.get('promotion')=='rollback':label='F0 - rolled back'
            if stage in ['source','neutral']:
                aligned=temp/(stage+'-aligned.wav')
                plan={'sampleRate':RATE,'frames':duration*24,'phrases':[{'startSample':0,'endSample':duration*24}]}
                align_phrases([path],plan,aligned);path=aligned
            mixed=output/(stage+'-mix.wav');mix_english(path,bed,duration,mixed)
            variants.append((stage,label,mixed))
        videos=[]
        for i,(stage,label,audio) in enumerate(variants):
            (temp/'label.txt').write_text(label,encoding='utf-8')
            path=output/(stage+'.mp4')
            ffmpeg(['-ss',str(start),'-i',report['source'],'-i',str(audio),'-t',str(duration/1000),
                '-map','0:v:0','-map','1:a:0','-vf',"scale=w='min(1280,iw)':h=-2,drawtext=textfile=label.txt:fontcolor=white:fontsize=24:box=1:boxcolor=black@0.7:x=20:y=20",
                '-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(path)],cwd=temp)
            videos.append(path);metadata.append({'stage':stage,'label':label,'video':path.name,'sourceStartMs':cue['startMs'],'durationMs':duration})
        playlist=temp/'concat.txt';playlist.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in videos))
        ffmpeg(['-f','concat','-safe','0','-i',str(playlist),'-c:v','copy','-c:a','aac','-t',str(duration/1000*len(videos)),'-movflags','+faststart',str(output/'ab-comparison.mp4')],timeout=180)
    write_ab_page(output,metadata)
    shutil.copy(folder/'reference.wav',output/'reference.wav')
    (output/'ab-manifest.json').write_text(json.dumps({'pilot':str(folder),'engine':args.engine,'seed':args.seed,'comparisons':metadata,'limitations':'Audio envelope/F0 and draft visemes do not prove phonetic lip synchrony. F0 variants marked rolled back are retained for review only.'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['pilot','background','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--engine',choices=['qwen','chatterbox'],required=True);p.add_argument('--seed',type=int,required=True);package(p.parse_args())
