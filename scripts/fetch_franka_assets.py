"""Mirror the official Franka USD dependencies while preserving relative paths."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen
import json,re,time
from pxr import Sdf
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/'
ROOT=Path(__file__).resolve().parents[1]/'assets/isaac_5_1'
initial='Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'
seen=set();pending={BASE+initial};manifest=[]
def fetch(url):
 relative=url[len(BASE):];path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  for retry in range(3):
   try:
    with urlopen(url,timeout=90) as response:path.write_bytes(response.read())
    break
   except Exception:
    if retry==2:raise
    time.sleep(2)
 refs=[]
 if path.suffix in ['.usd','.usda','.usdc']:
  layer=Sdf.Layer.FindOrOpen(str(path));assert layer,path
  for ref in re.findall(r'@([^@\n]*)@',layer.ExportToString()):
   if not ref or any(c in ref for c in ['"','{','}','=']):continue
   if ref.endswith('.mdl') and '/' not in ref:continue
   resolved=urljoin(url,ref)
   if resolved.startswith(BASE):refs.append(resolved)
 return relative,path.stat().st_size,refs
with ThreadPoolExecutor(max_workers=8) as pool:
 while pending:
  urls=sorted(pending-seen);pending=set();seen.update(urls)
  for relative,size,refs in pool.map(fetch,urls):
   manifest.append({'path':relative,'bytes':size});pending.update(set(refs)-seen)
   print(relative,size,flush=True)
(ROOT/'franka_manifest.json').write_text(json.dumps(manifest,indent=2))
print('DONE',len(manifest),sum(x['bytes'] for x in manifest),flush=True)
