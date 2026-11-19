import test from 'node:test';
import assert from 'node:assert/strict';
import {layoutPipeline, changePipelineDependency} from '../web/studio-pipeline-model.js';
const plan={nodes:[{id:'source',editable:false,dependsOn:[]},{id:'report',editable:false,dependsOn:['source']},{id:'a',editable:true,enabled:true,dependsOn:['report']},{id:'b',editable:true,enabled:true,dependsOn:['report']}],dependencies:{}};
test('mod order changes while required native boundary stays and source plan is immutable',()=>{
 const edges=changePipelineDependency(plan,{},'b','a',true);const graph=layoutPipeline(plan,edges);
 assert.deepEqual(graph.nodes.find(n=>n.id==='b').dependsOn,['report','a']);
 assert.ok(graph.nodes.find(n=>n.id==='b').column>graph.nodes.find(n=>n.id==='a').column);
 assert.deepEqual(plan.dependencies,{});
 assert.deepEqual(changePipelineDependency(plan,edges,'b','a',false),{});
});
test('cycle, self-link, native mutation, missing node and unavailable parent are rejected',()=>{
 const edges=changePipelineDependency(plan,{},'b','a',true);
 assert.throws(()=>changePipelineDependency(plan,edges,'a','b',true),/cycle/);
 assert.throws(()=>changePipelineDependency(plan,{},'a','a',true));
 assert.throws(()=>changePipelineDependency(plan,{},'report','a',true));
 assert.throws(()=>layoutPipeline({...plan,nodes:[...plan.nodes,{id:'x',dependsOn:['missing']}]}));
 assert.throws(()=>changePipelineDependency({...plan,nodes:plan.nodes.map(n=>n.id==='a'?{...n,enabled:false}:n)}, {},'b','a',true));
});
