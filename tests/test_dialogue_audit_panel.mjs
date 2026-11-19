import test from 'node:test';
import assert from 'node:assert/strict';
import {auditTime, mountDialogueAudit} from '../web/dialogue-audit.js';

/** Model only the panel's DOM protocol; no browser geometry or AT is inferred. */
class Element {
  constructor(tag) {
    this.tag=tag; this.children=[]; this.dataset={}; this.attributes={}; this.listeners={};
    this.styles={}; this.style={setProperty:(name,value)=>this.styles[name]=value};
  }
  setAttribute(name,value) {this.attributes[name]=value;}
  getAttribute(name) {return this.attributes[name];}
  hasAttribute(name) {return Object.hasOwn(this.attributes,name);}
  removeAttribute(name) {delete this.attributes[name];}
  addEventListener(name,callback) {this.listeners[name]=callback;}
  append(...children) {this.children.push(...children);}
  prepend(...children) {this.children.unshift(...children);}
  replaceChildren(...children) {this.children=children;}
}

/** Restore all process globals and expose mutable host report/source references. */
function setup(t, initialInventory=null) {
  const saved={document:globalThis.document,fetch:globalThis.fetch,setTimeout:globalThis.setTimeout,clearTimeout:globalThis.clearTimeout};
  globalThis.document={createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag)};
  t.after(()=>Object.assign(globalThis,saved));
  const owner=new Element('details'), seeks=[];
  let report={language:'ru',durationMs:90000,segments:[{startMs:100,endMs:1400,text:'Original <script>text</script>'}],speechTurns:[{startMs:100,endMs:1400,speaker:1}],sourceDigest:'b'.repeat(64),sceneEvidence:{secret:'not sent'},voiceProfiles:{secret:'not sent'}};
  if(initialInventory)report.dialogueAudit=initialInventory;
  let source=new Blob(['SOURCE MEDIA BYTES']);
  const panel=mountDialogueAudit(owner,{currentReport:()=>report,currentSource:()=>source,seek:ms=>seeks.push(ms)});
  panel.update(report);
  return {owner,panel,seeks,get report(){return report;},get source(){return source;},
    change(nextReport,nextSource=new Blob(['NEW SOURCE'])){report=nextReport;source=nextSource;panel.update(report);},
    inspect:()=>owner.children[1].children[0].onclick(),retry:()=>owner.children[1].children[1].onclick(),
    cancel:()=>owner.children[1].children[2].onclick(),status:()=>owner.children[2],list:()=>owner.children[3],more:()=>owner.children[4]};
}

/** Produce bounded, distinct findings so pagination and stale results are observable. */
function inventory(count=41,prefix='issue') {
  return {status:'complete',summary:{candidateCount:count},candidates:Array.from({length:count},(_,i)=>({id:prefix+i,startMs:100+i*1500,endMs:1400+i*1500,reasons:[prefix+' reason']}))};
}

/** Encode the actual NDJSON response framing consumed by readAnalysisStream. */
function response(value) {return new Response(JSON.stringify({result:value})+'\n');}

/** Traverse created nodes without treating text as parsed markup. */
function descendants(node) {return [node,...node.children.flatMap(descendants)];}

test('pagination is bounded, seeks source milliseconds, and resets for a new inventory',t=>{
  const view=setup(t,inventory());
  assert.equal(view.list().children.length,20);
  view.more().onclick();assert.equal(view.list().children.length,40);
  view.more().onclick();assert.equal(view.list().children.length,41);assert.equal(view.more().hidden,true);
  view.list().children[2].children[0].onclick();assert.deepEqual(view.seeks,[3100]);
  view.change({...view.report,dialogueAudit:inventory(25,'new')});
  assert.equal(view.list().children.length,20);assert.equal(view.more().hidden,false);
  assert.equal(auditTime(1999),'0:01.99');
});

test('upload Blob preserves exact source bytes, uses speech allowlist, and keeps hostile text inert',async t=>{
  const view=setup(t);
  view.source.arrayBuffer=()=>{throw Error('Must not copy the source Blob to JS RAM');};
  const raw=view.report.segments[0].text;
  const value=inventory(1,'<img src=x onerror=alert(1)>');
  value.repairs={suggestions:[{candidateId:value.candidates[0].id,startMs:100,text:'<script>unverified draft</script>'}]};
  let request;
  globalThis.fetch=async(url,options)=>{request={url,...options};return response(value);};
  await view.inspect();
  assert.ok(request.body instanceof Blob);assert.match(request.url,/rerun=0/);
  const bytes=new Uint8Array(await request.body.arrayBuffer()),size=new DataView(bytes.buffer).getUint32(0);
  const metadata=JSON.parse(new TextDecoder().decode(bytes.subarray(4,4+size)));
  assert.deepEqual(Object.keys(metadata).sort(),['durationMs','language','segments','sourceDigest','speechTurns']);
  assert.equal(new TextDecoder().decode(bytes.subarray(4+size)),'SOURCE MEDIA BYTES');
  assert.equal(view.report.segments[0].text,raw);
  const nodes=descendants(view.owner);
  assert.equal(nodes.some(node=>['script','img'].includes(node.tag)),false);
  const draft=nodes.find(node=>node.tag==='details'&&node!==view.owner);
  assert.equal(draft.styles.flex,'1 1 100%');
  assert.match(draft.children[1].textContent,/<script>unverified draft<\/script>/);
});

test('retry sends rerun=1 and labels proposals without applying them',async t=>{
  const view=setup(t,inventory(1));
  let url;
  const value=inventory(1);value.repairs={suggestions:Array.from({length:25},(_,i)=>({candidateId:'issue0',startMs:100,text:'Draft '+i}))};
  globalThis.fetch=async(path)=>{url=path;return response(value);};
  await view.retry();assert.match(url,/rerun=1/);
  assert.equal(view.list().children[0].children[2].children.length,21);
  assert.equal(view.report.segments[0].text,'Original <script>text</script>');
  assert.match(view.list().children[0].children[2].children[0].textContent,/требуют проверки/);
});

test('backend timeout, failed and cancelled states retain candidates and specific safe diagnostics',t=>{
  const view=setup(t);
  for(const state of ['timeout','failed','cancelled']){
    view.panel.update({dialogueAudit:{...inventory(1),status:state,error:'Specific <b>'+state+'</b>'}});
    assert.equal(view.list().children.length,1);assert.match(view.status().textContent,new RegExp(state));
    assert.equal(view.status().children.length,0);
  }
});

test('empty and unavailable states distinguish successful absence from missing evidence',t=>{
  const view=setup(t,inventory(0));assert.match(view.status().textContent,/не найдено/);
  view.panel.update({dialogueAudit:{status:'unavailable',message:'Measured turns required'}});
  assert.equal(view.status().textContent,'Measured turns required');assert.equal(view.more().hidden,true);
});

test('client deadline aborts a stalled wait and preserves the previous inventory',async t=>{
  const view=setup(t,inventory(1));let deadline,cleared=false;
  globalThis.setTimeout=(callback,ms)=>{assert.equal(ms,190000);deadline=callback;return 42;};
  globalThis.clearTimeout=id=>{if(id===42)cleared=true;};
  globalThis.fetch=(url,{signal})=>new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true}));
  const pending=view.inspect();assert.equal(view.owner.children[1].children[0].disabled,true);
  deadline();await pending;
  assert.match(view.status().textContent,/Истёк лимит ожидания/);assert.equal(view.list().children.length,1);
  assert.equal(view.owner.children[1].children[0].disabled,false);assert.equal(cleared,true);
});

test('cancel stops waiting, retains old findings and restores retry controls',async t=>{
  const view=setup(t,inventory(1));
  globalThis.fetch=(url,{signal})=>new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true}));
  const pending=view.inspect();view.cancel();await pending;
  assert.match(view.status().textContent,/Ожидание прервано/);assert.equal(view.list().children.length,1);
  assert.equal(view.owner.children[1].children[2].hidden,true);
});

test('reset frees the wait immediately and an old completion cannot repaint a new source',async t=>{
  const view=setup(t);let oldResolve,calls=0,oldSignal;
  globalThis.fetch=(url,options)=>{calls++;if(calls===1){oldSignal=options.signal;return new Promise(resolve=>oldResolve=resolve);}return Promise.resolve(response(inventory(1,'new')));};
  const old=view.inspect();view.panel.update(null);assert.equal(oldSignal.aborted,true);assert.equal(view.owner.hidden,true);
  const next={segments:[],speechTurns:[],durationMs:90000,language:'ru'};
  view.change(next);await view.inspect();assert.equal(calls,2);
  oldResolve(response(inventory(41,'old')));await old;
  assert.equal(next.dialogueAudit.candidates[0].id,'new0');assert.equal(view.list().children.length,1);
  assert.equal(view.list().children[0].children[1].textContent,'new reason');
});

test('dispose aborts waiting, detaches panel and ignores late work/update',async t=>{
  const view=setup(t);let resolve,signal;
  globalThis.fetch=(url,options)=>{signal=options.signal;return new Promise(done=>resolve=done);};
  const pending=view.inspect();view.panel.dispose();assert.equal(signal.aborted,true);assert.equal(view.owner.children.length,0);
  resolve(response(inventory(1)));await pending;view.panel.update(view.report);
  assert.equal(view.owner.children.length,0);assert.equal(view.owner.hidden,true);
});

test('HTTP errors preserve findings and show literal safe text',async t=>{
  const view=setup(t,inventory(1));
  globalThis.fetch=async()=>new Response(JSON.stringify({error:'<img src=x> synthetic denied'}),{status:503});
  await view.inspect();assert.equal(view.list().children.length,1);assert.match(view.status().textContent,/<img src=x>/);
  assert.equal(descendants(view.owner).some(node=>node.tag==='img'),false);
});
