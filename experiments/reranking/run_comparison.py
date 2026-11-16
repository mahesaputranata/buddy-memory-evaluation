"""One command: Llama only, 15 reranking-OFF contexts + 15 ON contexts."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import ollama_eval_core as core


def load_answers(folder):
    return [json.loads(line) for line in (folder/'answers.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


def answer_text(row):
    try:
        return json.loads(row['raw_answer']).get('answer',row['raw_answer'])
    except (json.JSONDecodeError,AttributeError):
        return row['raw_answer']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='http://127.0.0.1:11434')
    parser.add_argument('--output',default=None)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--on-first',action='store_true',help='Reverse condition order for an optional order-effect check')
    args=parser.parse_args(argv)
    here=Path(__file__).resolve().parent
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
    output=Path(args.output or ('Llama_reranking_results_'+stamp+'_'+str(time.time_ns())[-6:])).resolve()
    output.mkdir(parents=True,exist_ok=False)
    order=['on','off'] if args.on_first else ['off','on']
    datasets={label:json.loads((here/f'cases_{label}.json').read_text(encoding='utf-8')) for label in ['off','on']}
    assert [c['qid'] for c in datasets['off']['cases']]==[c['qid'] for c in datasets['on']['cases']]
    assert len(datasets['off']['cases'])==15
    manifest={'experiment':'Llama_frozen_reranking_OFF_ON_answer_comparison',
              'model':'llama3.2:latest','condition_order':order,
              'data_provenance':{label:datasets[label]['provenance'] for label in ['off','on']},
              'runner_sha256':hashlib.sha256((here/'ollama_eval_core.py').read_bytes()).hexdigest(),
              'decision':'Keep OFF if ON provides no meaningful answer-quality benefit; no universal model claim.',
              'limitations':['15 development questions, one answer per case/condition',
                    'Frozen B6 contexts; local retrieval/reranking latency is not measured',
                    'Generation latency can include repeated model-loading overhead',
                    'No automatic semantic correctness grading; review actual answers']}
    (output/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for label in order:
        print('\n=== RERANKING '+label.upper()+' CONTEXT: 15 LLAMA ANSWERS ===',flush=True)
        command=['--host',args.host,'--models','llama3.2:latest',
                 '--conditions','retrieved','--cases',str(here/f'cases_{label}.json'),
                 '--output',str(output/label)]
        if args.dry_run:command.append('--dry-run')
        core.main(command)
    if args.dry_run:
        print('Dry run complete; no model requests. Inputs:',output)
        return output

    off={r['qid']:r for r in load_answers(output/'off')}
    on={r['qid']:r for r in load_answers(output/'on')}
    reviews=[]
    lines=['# Llama: reranking OFF versus ON',
           '', 'These are answers to the same 15 questions with the same model/settings. Only the frozen evidence selection/order changes.',
           '', '**Do not grade by fluency or JSON format alone.** Check requested facts, completeness, source support, and uncertainty. Latency here is answer-request time, not live reranking cost.', '']
    for case in datasets['off']['cases']:
        qid=case['qid'];a=off[qid];b=on[qid]
        contexts={label:next(c for c in datasets[label]['cases'] if c['qid']==qid)['contexts']['retrieved']
                  for label in ['off','on']}
        row={'qid':qid,'question':case['question'],
             'off_answer':answer_text(a),'on_answer':answer_text(b),
             'same_answer_text':' '.join(answer_text(a).split())==' '.join(answer_text(b).split()),
             'reference_answer':case['evaluation_only']['reference_answer'],
             'off_client_ms':a['client_elapsed_ms'],'on_client_ms':b['client_elapsed_ms'],
             'off_load_ms':a['load_ms'],'on_load_ms':b['load_ms'],
             'off_error':a['error'],'on_error':b['error'],
             'off_evidence':'\n\n'.join(e['text'] for e in contexts['off']),
             'on_evidence':'\n\n'.join(e['text'] for e in contexts['on']),
             'off_correct_complete_0_or_1':'','on_correct_complete_0_or_1':'',
             'off_unsupported_claims':'','on_unsupported_claims':'',
             'verdict_on_better_off_better_or_tie':'','review_note':''}
        reviews.append(row)
        lines.extend(['## '+case['question'],'', '**OFF:** '+answer_text(a),'',
                      '**ON:** '+answer_text(b),'',
                      '**Reference answer:** '+case['evaluation_only']['reference_answer'],'',
                      '**Raw OFF output:**','```json',a['raw_answer'],'```','',
                      '**Raw ON output:**','```json',b['raw_answer'],'```','',
                      '**Errors:** OFF='+a['error']+'; ON='+b['error'],'',
                      '**OFF evidence:**','```text',row['off_evidence'],'```','',
                      '**ON evidence:**','```text',row['on_evidence'],'```',''])
    core.write_csv(output/'paired_answer_review.csv',reviews)
    (output/'READ_RESULTS_FIRST.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\nFOR REVIEW: ',output/'READ_RESULTS_FIRST.md')
    print('Blank correctness/verdict fields are deliberate: answer quality needs evidence review.')
    archive=Path(str(output)+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(output.rglob('*')):
            if file.is_file() and file.suffix in {'.json','.jsonl','.csv','.md'}:
                z.write(file,arcname=str(file.relative_to(output)))
    print('\nUPLOAD THIS RESULTS ZIP:',archive,flush=True)
    return output


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(1)
