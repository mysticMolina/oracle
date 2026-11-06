"""One accepted step of the checked planar case; target duration is unchanged."""
from pathlib import Path
import json
from oracle_py.campaign import run_campaign
root=Path(__file__).resolve().parents[1]
config=json.loads((root/'fixtures/matched_B200_box26r.json').read_text())
output=root/'output/quickstart'
if output.exists():raise SystemExit('Use a new output directory; existing results are preserved.')
result=run_campaign(config,output,stop_after=1,checkpoint_every=1)
assert result['step']==1 and result['status']=='partial',result
print('Output:',output)
print('Accepted steps:',result['step'],'of',result['stepsRequested'])
