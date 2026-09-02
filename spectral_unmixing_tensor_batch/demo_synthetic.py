# -*- coding: utf-8 -*-
"""人工数据集成演示，不能当作真实WorldStrat提取精度。
用法：python demo_synthetic.py --scripts "五个旧脚本所在目录" --output "新演示目录"
不需要网络或真实遥感数据。调用五个真实旧模块，而非伪造其main返回值。
"""
from pathlib import Path
import argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin
from spectral_unmixing_pipeline import Config, ClassSpec, CLASSES, main
from unmixing_core import SolverConfig

BANDS12=('B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12')


def create_synthetic_scene(directory: Path, level='L2A', no_building=False):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    # 完全人为设计的光谱，仅用来检验条件分支和数值求解。
    E=np.array([
        [.018,.025,.060,.025,.120,.300,.370,.400,.380,.360,.180,.090],
        [.025,.028,.040,.020,.015,.010,.009,.008,.007,.006,.005,.003],
        [.080,.130,.180,.220,.240,.260,.280,.300,.310,.290,.400,.340],
        [.720,.780,.850,.820,.790,.760,.740,.720,.650,.440,.120,.080],
        [.180,.220,.250,.280,.290,.300,.310,.280,.300,.290,.420,.350],
    ]).T
    bands=list(BANDS12)
    if level=='L1C':
        bands.insert(10,'B10')
        E=np.insert(E,10,np.array([.003,.002,.004,.007,.005]),axis=0)
    h,w=60,120
    rng=np.random.default_rng(7301)
    A=np.zeros((5,h,w))
    # 上半部分：五个有来源支持的大型纯区域；下半部分：随机1—3类混合。
    for c in range(5):
        effective=2 if no_building and c==4 else c
        A[effective,:36,24*c:24*(c+1)]=1
    allowed=np.arange(4 if no_building else 5)
    for r in range(36,h):
        for col in range(w):
            count=int(rng.integers(1,4))
            ids=rng.choice(allowed,count,replace=False)
            A[ids,r,col]=rng.dirichlet(np.ones(count))
    cube=(E@A.reshape(5,-1)).reshape(len(bands),h,w)
    cube+=rng.normal(0,.00012,cube.shape)
    cube[:,0,0]=np.nan
    path=directory/f'synthetic_{level}.tif'
    profile=dict(driver='GTiff',width=w,height=h,count=len(bands),dtype='float32',
                 crs='EPSG:4326',transform=from_origin(110,30,.0001,.0001),nodata=np.nan)
    with rasterio.open(path,'w',**profile) as dst:
        dst.write(cube.astype('float32'))
        dst.descriptions=tuple(bands)
    def mask_file(name,mask):
        dest=directory/f'{name}.tif'
        p={**profile,'count':1,'dtype':'uint8','nodata':None}
        with rasterio.open(dest,'w',**p) as dst:
            dst.write(mask.astype('uint8'),1)
        return dest
    bare=np.zeros((h,w),bool);bare[:36,48:72]=True
    roof=np.zeros((h,w),bool)
    if not no_building:roof[:36,96:120]=True
    clear=np.ones((h,w),bool);clear[-2:,-2:]=False
    masks={'bare':mask_file('bare_support',bare),'building':mask_file('roof_support',roof),
           'clear':mask_file('clear_mask',clear)}
    np.savez_compressed(directory/'synthetic_truth.npz',E=E,abundance=A,bands=np.asarray(bands))
    return path,masks,A,E


def demo_config(script_dir: Path, output: Path, level='L2A', plots=True):
    input_path,masks,A,E=create_synthetic_scene(output/'input',level)
    specs={c:ClassSpec(reviewed=True,max_candidates=2) for c in CLASSES}
    # reviewed=True只因人工构造数据有明确真值；不要照搬到真实影像。
    for c in CLASSES:
        specs[c].extraction_overrides={'max_clusters':2,'n_init':2,'silhouette_sample_size':250}
    specs['bare'].extraction_overrides.update(support_mask_path=masks['bare'])
    specs['building'].extraction_overrides.update(roof_support_mask_path=masks['building'],mode='ROOF_PRIOR')
    cfg=Config(input_path=input_path,script_dir=script_dir,output_root=output/'runs',
               data_confirmed=True,scale=1.,offset=0.,clear_mask_path=masks['clear'],classes=specs,
               make_plots=plots,make_html=plots,save_example_pixels=2 if plots else 0,
               solver=SolverConfig(selection_delta_mse=1e-8,ambiguity_delta_mse=1e-8,max_rmse=.01))
    return cfg,A,E


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scripts',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'synthetic_demo')
    parser.add_argument('--level',choices=['L2A','L1C'],default='L2A')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    cfg,truth,E=demo_config(args.scripts,args.output,args.level,not args.no_plots)
    data=main(cfg)
    valid=data['scene']['valid']
    expected=truth[:,valid].T
    error=np.abs(data['abundance']-expected)
    print('人工数据类别丰度平均绝对误差：',error.mean())
    print('这不是实际遥感数据的精度验证。')
