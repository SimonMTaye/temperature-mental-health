"""Frozen current-source palm control ladders and separate migration restrictions."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
import ast,gc,hashlib,json,re,sys,subprocess,warnings
from dataclasses import dataclass,field
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

HERE=Path(__file__).resolve().parent;TASK=HERE.parent;ROOT=HERE.parents[2]
REPO=ROOT/'repository_review_snapshot_2026-11-02';ACTIVE=ROOT/'analysis'
INPUT=ACTIVE/'data/generated/30_analysis_table_input.parquet'
FLAGS=ROOT/'meeting_robustness_2026-10-09/palm_identification/roster/person_migration_diagnostics.parquet'
ROSTER_META=FLAGS.parent/'roster_migration_metadata.json'
SHA='5b72cc567944f4b8e0cfad918727c40198fe73f6'
INPUT_SHA='91927f088843c2877b567f41bdd772046c7a6ca1bb4b640739a2eeed39e10577'
PROTOCOL_SHA='8cc260233daed71d6472ecc29e7254f369c8c0ee83e0c6fa1dd362bc978f125f'
P='palm_farmer_hh_ifls4';G='palm_price_gap_z';HEATS=['tmean_7d','wetbulb_7d']
OUT=HERE/'results';OUT.mkdir(parents=True,exist_ok=True)
for name in ['samples','vcov','design']:(OUT/name).mkdir(exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for chunk in iter(lambda:s.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def sample_hash(d):return hashlib.sha256('|'.join(sorted(d.pidlink.astype(str)+'_'+d.wave.astype(str))).encode()).hexdigest()
def json_native(value):
    if isinstance(value,dict):return {str(k):json_native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_native(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,np.floating,np.bool_)):return value.item()
    if isinstance(value,(pd.Series,pd.DataFrame)):return json_native(value.to_dict())
    return value if value is None or isinstance(value,(str,int,float,bool)) else str(value)
def nodes(path,env,names):
    found=set()
    for n in ast.parse(path.read_text(encoding='utf-8')).body:
        decl={n.name} if isinstance(n,(ast.FunctionDef,ast.ClassDef)) else set()
        if isinstance(n,ast.Assign):decl={t.id for t in n.targets if isinstance(t,ast.Name)}
        if decl&set(names):exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),env);found|=decl&set(names)
    assert found==set(names),(path,set(names)-found)
def preservation():
    paths=[x for x in (ACTIVE/'code').rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix not in ['.pyc','.pyo']]
    paths += [INPUT,ACTIVE/'data/generated/20_economic_exposures.parquet',FLAGS]
    for rel in ['code/library/specs.py','code/library/render.py','code/library/caching.py','code/analysis/tables_v2/table_b_temperature_and_shock_effects.py','code/data/20_build_economic_exposures.py','code/data/30_build_analysis_table_input.py','code/data/26_process_temperature_data.py']:paths.append(REPO/rel)
    for rec in json.loads(ROSTER_META.read_text())['fingerprints']:
        if '/raw/' in rec['path'].replace('\\','/'):
            p=ROOT/rec['path'];assert sha(p)==rec['sha256'];paths.append(p)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
def prune(d,fe):
    d=d.copy()
    while len(d):
        keep=pd.Series(True,index=d.index)
        for c in fe:keep &= d.groupby(c,observed=True)[c].transform('size').gt(1)
        if keep.all():return d
        d=d.loc[keep].copy()
    raise AssertionError('Empty sample')
def count(d):
    palm=d.loc[d[P].eq(1)]
    pairs=d.groupby('pidlink',observed=True).wave.nunique()
    return {'n':len(d),'people':d.pidlink.nunique(),'paired_people':int(pairs.eq(2).sum()),'household_waves':len(d[['hhid','wave']].drop_duplicates()),'districts':d.kabupaten_full_code.nunique(),'palm_rows':len(palm),'palm_people':palm.pidlink.nunique(),'palm_districts':palm.kabupaten_full_code.nunique(),'origin_provinces':d.origin_province.nunique(),'palm_origin_provinces':palm.origin_province.nunique(),'IFLS4_n':int(d.wave.eq('IFLS4').sum()),'IFLS5_n':int(d.wave.eq('IFLS5').sum()),'sample_hash':sample_hash(d)}
def main():
    assert pf.__version__=='0.50.1' and pd.__version__=='3.0.3'
    assert sha(INPUT)==INPUT_SHA and sha(TASK/'PROTOCOL.md')==PROTOCOL_SHA
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==SHA
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    before=preservation();(HERE/'preflight_manifest.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'files':before,'input_sha256':INPUT_SHA,'protocol_sha256':PROTOCOL_SHA,'commit':SHA},indent=2),encoding='utf-8')
    controls=['age','female','edu_yrs_nullable','married','widowed','divorced','ethnicity_nullable','religion_nullable']
    cols=['pidlink','wave','hhid','cesd_z',P,G,*HEATS,*controls,'month','year','province_code','gadm_fullcode','kabupaten_full_code']
    d=pd.read_parquet(INPUT,columns=cols);assert len(d)==60419 and not d.duplicated(['pidlink','wave']).any()
    d['canonical_row_id']=np.arange(len(d));d['ifls5']=d.wave.eq('IFLS5').astype(int)
    b=d.loc[d.wave.eq('IFLS4'),['pidlink','province_code','year','month']].rename(columns={'province_code':'origin_province','year':'baseline_year','month':'baseline_month'})
    d=d.merge(b,on='pidlink',how='left',validate='m:1');assert np.array_equal(d.canonical_row_id,np.arange(len(d)))
    for c in ['origin_province','baseline_year','baseline_month']:d[c]=d[c].astype('Int64').astype('string')
    d['month_year']=d.year.astype('Int64').astype('string')+'_'+d.month.astype('Int64').astype('string')
    pp=d[P].astype('Int64').astype('string');mm=d.month.astype('Int64').astype('string')
    d['palm_month']=pp+'_'+mm
    d['cohort_palm_wave']=d.baseline_year+'_'+d.baseline_month+'_'+pp+'_'+d.wave
    d['origin_wave']=d.origin_province+'_'+d.wave
    flags=pd.read_parquet(FLAGS,columns=['pidlink','mg18e_migrant','mg20c_any_move','nonmigrant_consistent','mg_interwave_scope','track_followup_hhid_agrees'])
    assert flags.pidlink.is_unique
    d=d.merge(flags,on='pidlink',how='left',validate='m:1',indicator=True);assert d['_merge'].eq('both').all();d=d.drop(columns='_merge')
    assert np.array_equal(d.canonical_row_id,np.arange(len(d)));d.index.name='canonical_row'
    env={'pd':pd,'np':np,'re':re,'dataclass':dataclass,'field':field,'__name__':__name__,'analysis_df':d}
    nodes(REPO/'code/library/render.py',env,['FORMULA_SPLIT_PATTERN','RegressionSpec','search_replace_term'])
    nodes(REPO/'code/library/specs.py',env,['CONTROLS_NULLABLE','CONTROLS','CONTROLS_PANEL','FE_NO_WAVE','MAIN_TEMP_MEASURE','update_formula_search_replace','shock_triple_diff_panel','palm_shock','palm_shock_panel'])
    nodes(REPO/'code/analysis/tables_v2/table_b_temperature_and_shock_effects.py',env,['PALM_PRICE_SHOCK'])
    assert [x.strip() for x in env['CONTROLS'].split('+')]==controls
    sources={'continuous_pooled':env['PALM_PRICE_SHOCK'].formula,'binary_panel':env['palm_shock_panel'].formula}
    frames={};plans=[];flow=[];aliases=[];ladder=[];migration=[];source_registry=[];migration_counts=[]
    labels={1:'Matched reference',2:'Palm × calendar-month FE',3:'Origin-province levels and heat slopes',4:'Both requested controls',5:'Also baseline-cohort × palm × wave levels',6:'Also origin-province × wave levels'}
    for design,source in sources.items():
        group=G if design=='continuous_pooled' else P;is_panel=design=='binary_panel'
        for heat in HEATS:
            tag=design+'_'+heat;literal=' '.join(source.replace('tmean_7d',heat).split())
            lhs,sourcefe=[x.strip() for x in literal.split('|')]
            source_fitted=lhs+(' - ifls5' if not is_panel else '')+' | '+sourcefe
            source_fe=['month_year','gadm_fullcode']+(['pidlink','ifls5'] if is_panel else [])
            need=['cesd_z',heat,group,*controls,'month','year','gadm_fullcode','kabupaten_full_code','pidlink']
            complete=d.dropna(subset=need);full=prune(complete,source_fe)
            eligible=d.dropna(subset=list(dict.fromkeys(need+[P,'origin_province','baseline_year','baseline_month','palm_month','cohort_palm_wave','origin_wave'])))
            common=prune(eligible,list(dict.fromkeys(source_fe+['palm_month','origin_province','cohort_palm_wave','origin_wave'])))
            assert set(common.index)<=set(full.index)
            if is_panel:assert full.groupby('pidlink').size().eq(2).all() and common.groupby('pidlink').wave.nunique().eq(2).all()
            levels=sorted(common.origin_province.unique(),key=int);ref=levels[0]
            for frame in [d,full,common]:
                for lev in levels[1:]:frame['origin_heat_'+lev]=frame.origin_province.eq(lev).astype(float)*frame[heat]
            frames[tag+'_source']=full;frames[tag+'_common']=common
            full.to_parquet(OUT/'samples'/(tag+'_source.parquet'),index=True);common.to_parquet(OUT/'samples'/(tag+'_common.parquet'),index=True)
            fullid=tag+'_source'
            plans.append({'model_id':fullid,'design':design,'heat':heat,'group':group,'role':'source_reference','ladder_row':0,'label':'Unrestricted source reference','frame_id':tag+'_source','literal_source_formula':literal,'conceptual_formula':literal,'fitted_formula':source_fitted,'origin_reference':ref,'full_source_input':True,'absorbed_terms':['ifls5 RHS (month-year FE)'] if not is_panel else [],**count(full)})
            source_registry.append({'design':design,'heat':heat,'model_id':fullid})
            sf={'design':design,'heat':heat,'source_complete':len(complete),'source_singletons':len(complete)-len(full),'source_n':len(full),'common_eligible':len(eligible),'common_singletons':len(eligible)-len(common),'common_n':len(common),'source_vs_common_same_keys':full.index.equals(common.index),'source_removed_to_common':len(full)-len(common),'source_hash':sample_hash(full),'common_hash':sample_hash(common),'origin_reference':ref};flow.append(sf)
            for row in range(1,7):
                modelid=tag+f'_ladder_{row:02d}'
                if row==1 and full.index.equals(common.index):
                    aliases.append({'alias':modelid,'model_id':fullid,'reason':'Exact same source formula and sample'});ladder.append({'design':design,'heat':heat,'ladder_row':row,'label':labels[row],'model_id':fullid});continue
                addfe=[];extra=[];absorbed=[];conceptfe=[]
                if row in [2,4,5,6]:addfe.append('palm_month');conceptfe.append('palm_month')
                if row>=3:
                    extra=['origin_heat_'+lev for lev in levels[1:]];conceptfe.append('origin_province')
                    if not is_panel and row!=6:addfe.append('origin_province')
                    elif is_panel:absorbed.append('New origin-province levels absorbed by existing person FE')
                if row>=5:addfe.append('cohort_palm_wave');conceptfe.append('cohort_palm_wave')
                if row==6:addfe.append('origin_wave');conceptfe.append('origin_wave');absorbed.append('Separate added origin-province levels contained in origin-province × wave FE')
                actual_lhs=lhs
                if not is_panel:actual_lhs+=' - ifls5';absorbed.append('ifls5 RHS absorbed by month-year FE')
                if is_panel and row>=5:actual_lhs+=f' - {P}:ifls5';absorbed.append('Palm × wave level absorbed by baseline-cohort × palm × wave FE')
                fitted=actual_lhs+(' + '+' + '.join(extra) if extra else '')+' | '+sourcefe+(' + '+' + '.join(addfe) if addfe else '')
                conceptual=lhs+(' + origin-province-specific heat slopes' if extra else '')+' | '+sourcefe+(' + '+' + '.join(conceptfe) if conceptfe else '')
                plans.append({'model_id':modelid,'design':design,'heat':heat,'group':group,'role':'common_ladder','ladder_row':row,'label':labels[row],'frame_id':tag+'_common','literal_source_formula':literal,'conceptual_formula':conceptual,'fitted_formula':fitted,'origin_reference':ref,'full_source_input':False,'absorbed_terms':absorbed,**count(common)})
                ladder.append({'design':design,'heat':heat,'ladder_row':row,'label':labels[row],'model_id':modelid})
            if is_panel:
                migration.append({'heat':heat,'migration_sample':'reference','model_id':fullid})
                for name,mask in [('nonmigrant',full.mg18e_migrant.eq(0)),('strict',full.nonmigrant_consistent.eq(1))]:
                    before_prune=full.loc[mask];f=prune(before_prune,source_fe);assert f.groupby('pidlink').size().eq(2).all()
                    mid=tag+'_'+name;frames[mid]=f;f.to_parquet(OUT/'samples'/(mid+'.parquet'),index=True)
                    plans.append({'model_id':mid,'design':design,'heat':heat,'group':group,'role':'migration','ladder_row':None,'label':name,'frame_id':mid,'literal_source_formula':literal,'conceptual_formula':literal,'fitted_formula':source_fitted,'origin_reference':None,'full_source_input':False,'absorbed_terms':[],**count(f)})
                    migration.append({'heat':heat,'migration_sample':name,'model_id':mid})
                    flow.append({'design':design,'heat':heat,'migration_sample':name,'restricted_before_prune':len(before_prune),'n':len(f),'removed_singletons':len(before_prune)-len(f)})
                for subset,data in [('reference',full),('nonmigrant',frames[tag+'_nonmigrant']),('strict',frames[tag+'_strict'])]:
                    for palm,block in data.loc[data.wave.eq('IFLS4')].groupby(P,observed=True):
                        for flag in ['mg18e_migrant','mg20c_any_move','nonmigrant_consistent']:migration_counts.append({'heat':heat,'sample':subset,'palm_proxy':int(palm),'measure':flag,'people':len(block),'yes':int(block[flag].eq(1).sum()),'no':int(block[flag].eq(0).sum()),'unknown':int(block[flag].isna().sum())})
    assert 28<=len(plans)<=32 and len(ladder)==24 and len(migration)==6
    pd.DataFrame(plans).to_json(OUT/'planned_models.json',orient='records',indent=2)
    pd.DataFrame(plans).drop(columns='absorbed_terms').to_csv(OUT/'planned_models.csv',index=False)
    for filename,rows in [('sample_flow',flow),('aliases',aliases),('ladder_registry',ladder),('migration_registry',migration),('source_registry',source_registry),('migration_counts',migration_counts)]:pd.DataFrame(rows).to_csv(OUT/(filename+'.csv'),index=False)
    print(pd.DataFrame(flow).to_string(index=False),flush=True);print('Planned unique fits: '+str(len(plans)),flush=True)
    if '--prepare-only' in sys.argv:return
    results=[];allterms=[];diagnostics=[]
    for plan in plans:
        expected=frames[plan['frame_id']];heat=plan['heat']
        frame=d.copy() if plan['full_source_input'] else expected.copy()
        # Slopes are regenerated for this heat; d is shared across the preparation loop.
        if plan['origin_reference'] is not None:
            for lev in sorted(expected.origin_province.dropna().unique(),key=int):frame['origin_heat_'+lev]=frame.origin_province.eq(lev).astype(float)*frame[heat]
        required=set(re.findall(r'\b[A-Za-z_]\w*\b',plan['fitted_formula']))&set(frame.columns)
        frame=frame[list(dict.fromkeys(['pidlink','wave','hhid','canonical_row_id','kabupaten_full_code',*sorted(required)]))].copy()
        with warnings.catch_warnings(record=True) as caught:model=pf.feols(plan['fitted_formula'],data=frame,vcov={'CRV1':'kabupaten_full_code'})
        actual_ids=model._data.canonical_row_id.to_numpy();assert np.array_equal(actual_ids,expected.canonical_row_id.to_numpy())
        assert model._N==len(expected)
        tidy=model.tidy();names=list(tidy.index);b=tidy.Estimate.to_numpy();v=np.asarray(model._vcov);x=np.asarray(model._X)
        target=next(n for n in names if set(n.split(':'))=={plan['group'],'ifls5',heat})
        j=names.index(target);norm=np.linalg.norm(x,axis=0);assert (norm>0).all()
        rank=int(np.linalg.matrix_rank(x/norm));assert rank==x.shape[1],(plan['model_id'],rank,x.shape)
        others=np.delete(x,j,axis=1);partial=x[:,j]-others@np.linalg.lstsq(others,x[:,j],rcond=None)[0]
        fraction=float(np.linalg.norm(partial)/np.linalg.norm(x[:,j]));assert fraction>1e-8,'Target numerically absorbed'
        df=float(model._df_t);assert df==expected.kabupaten_full_code.nunique()-1
        r=tidy.loc[target];crit=stats.t.ppf(.975,df)
        assert np.allclose([r['2.5%'],r['97.5%']],[r['Estimate']-crit*r['Std. Error'],r['Estimate']+crit*r['Std. Error']])
        assert np.isclose(r['Pr(>|t|)'],2*stats.t.sf(abs(r['Estimate']/r['Std. Error']),df))
        record={**plan,'term':target,'estimate':float(r['Estimate']),'se':float(r['Std. Error']),'p':float(r['Pr(>|t|)']),'ci_low':float(r['2.5%']),'ci_high':float(r['97.5%']),'inference_df':df,'rank':rank,'columns':x.shape[1],'scaled_condition':float(np.linalg.cond(x/norm)),'target_residual_fraction':fraction,'outcome_mean':float(expected.cesd_z.mean())};results.append(record)
        tt=tidy.reset_index();tt['model_id']=plan['model_id'];allterms.append(tt)
        pd.DataFrame(v,index=names,columns=names).to_csv(OUT/'vcov'/(plan['model_id']+'.csv'))
        np.savez_compressed(OUT/'design'/(plan['model_id']+'.npz'),X=x,Y=np.asarray(model._Y).reshape(-1),residual=np.asarray(model._u_hat).reshape(-1),beta=b,vcov=v,coef_names=np.array(names),row_index=actual_ids,cluster=expected.kabupaten_full_code.astype(str).to_numpy(dtype=str))
        native_ssc={key:json_native(getattr(model,key)) for key in ['_ssc','_ssc_dict','_k','_df_k','_k_fe','_G','_df_t','_fixef_sizes','_fixef','_fixef_dict','_N'] if hasattr(model,key)}
        diagnostics.append({'model_id':plan['model_id'],'warnings':[str(w.message) for w in caught],'estimator_dropped_terms':list(getattr(model,'_collin_vars',[])),'analytical_absorptions':plan['absorbed_terms'],'rank':rank,'columns':x.shape[1],'target_residual_fraction':fraction,'sample_hash':sample_hash(expected),'native_ssc':native_ssc})
        rr=pd.DataFrame(results);rr.to_csv(OUT/'models.csv',index=False);pd.concat(allterms,ignore_index=True).to_csv(OUT/'all_model_terms.csv',index=False)
        for filename,registry in [('ladder_results',ladder),('migration_results',migration),('source_reference_results',source_registry)]:
            pd.DataFrame(registry).merge(rr.drop(columns=[c for c in ['design','heat','ladder_row','label'] if c in pd.DataFrame(registry).columns]),on='model_id',how='inner',validate='m:1').to_csv(OUT/(filename+'.csv'),index=False)
        (OUT/'model_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
        print(json.dumps({k:record[k] for k in ['model_id','n','estimate','se','p','target_residual_fraction']}),flush=True)
        del model,frame,x,others,partial,tidy;gc.collect()
    after=preservation();assert before==after
    (HERE/'postflight_manifest.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'unchanged':True,'files':after},indent=2),encoding='utf-8')
    meta={'utc':datetime.now(timezone.utc).isoformat(),'commit':SHA,'input_sha256':sha(INPUT),'protocol_sha256':sha(TASK/'PROTOCOL.md'),'runner_sha256':sha(Path(__file__)),'pyfixest':pf.__version__,'pandas':pd.__version__,'controls':env['CONTROLS'],'cluster':'Current district CRV1; source defaults','unique_models':len(results),'ladder_rows':24,'migration_rows':6,'source_reference_rows':4,'aliases':aliases,'files_unchanged':before==after,'source_formulas':sources,'sample_flow':flow,'diagnostics':diagnostics,'notes':['Origin province is baseline province_code, not province_full_code. Stored continuous index is not restandardized.','Full source FE labels retained; algebraic omissions affect exactly absorbed terms only.','Region heat slopes omit the smallest observed origin-province category; no province-by-wave heat slopes.','Row6 origin-wave FE contain origin levels, so no separate redundant added origin label; panel person FE already contain origin levels.','No fixed-length ID rule, no new age-group interactions, no deferred income repair or source edits.','Controlling levels and fixed slopes does not eliminate all changing palm-specific confounders; migration restrictions change the sample.']}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('All fixed palm checks complete; protected files unchanged.',flush=True)

if __name__=='__main__':main()
