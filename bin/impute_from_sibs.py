#!/well/kong/users/wiw765/anaconda2/bin/python
import numpy as np
import numpy.ma as ma
import argparse, h5py
from pysnptools.snpreader import Bed

parser = argparse.ArgumentParser()
parser.add_argument('chr',type=int,help='Which chromosome (integer)')
parser.add_argument('ibd',type=str,help='IBD file')
parser.add_argument('genotypes',type=str,help='Genotypes in .bed format')
parser.add_argument('ped',type=str,help='Pedigree file with siblings sharing family ID')
parser.add_argument('out',type=str,help='Prefix of hdf5 output of imputed parental genotypes')
parser.add_argument('--king',action='store_true',default=False,help='IBD segments file in KING format (default 23andme)')
parser.add_argument('--start', type=int,
                    help='Start index of SNPs to perform imputation for in genotype file (starting at zero)',
                    default=0)
parser.add_argument('--end', type=int,
                    help='End index of SNPs to perform imputation for in genotype file (goes from 0 to (end-1)',
                    default=None)
args=parser.parse_args()
chr=args.chr

def pair_index(i,j):
    if j > i:
        return j*(j-1)/2+i
    else:
        return i*(i-1)/2+j

def impute_gt_2(ibd,sib1,sib2,f):
    if ibd==0:
        return sib1+sib2
    elif ibd==2:
        return sib1+2*f
    elif ibd==1:
        sibsum = sib1+sib2
        if sibsum==0:
            return f
        elif sibsum==1:
            return 1+f
        elif sibsum==2:
            return 1+2*f
        elif sibsum==3:
            return 2+f
        elif sibsum==4:
            return 3+f

def impute_gt(sibgts,ibd,f):
    n = len(sibgts)
    if not len(ibd) == n*(n-1)/2:
        raise(ValueError('Length of IBD state vector does not match number of sib pairs'))
    # If pair, do imputation
    if n < 2:
        raise(ValueError('Too few siblings'))
    if n == 2:
        return impute_gt_2(ibd,sibgts[0],sibgts[1],f)
    # Check for IBD_0
    imputed = 0
    count = 0
    ibd_0_count = 0
    for i in xrange(1,n):
        for j in xrange(0,i):
            if ibd[count]==0:
                imputed += sibgts[i]+sibgts[j]
                ibd_0_count += 1
            count += 1
    if ibd_0_count>0:
        return imputed/float(ibd_0_count)
    else:
        # Check if all IBD_1 (implies IBD inference error)
        if np.sum(ibd==1) == ibd.shape[0]:
            # Average over pairs
            imputed = 0
            for i in xrange(1,n):
                for j in xrange(0,i):
                    imputed += impute_gt_2(1,sibgts[i],sibgts[j],f)
            return imputed/float(ibd.shape[0])
        # Check if all IBD_2
        # return most common genotype + imputation from pop. freq.
        if np.sum(ibd==2) == ibd.shape[0]:
            return np.argmax(np.bincount(sibgts))+2*f
        # Collapse IBD_2
        ibd_2_count = np.zeros((n),dtype=int)
        remove = []
        count = 0
        for i in xrange(1, n):
            for j in xrange(0, i):
                if ibd[count] == 2:
                    ibd_2_count[np.array([i,j])] += 1
                count += 1
        remove = np.where(ibd_2_count==np.max(ibd_2_count))[0][0]
        keep = np.array([x for x in xrange(0,n)])
        keep = np.delete(keep,remove)
        sibgts = sibgts[keep]
        new_ibd = np.zeros((keep.shape[0]*(keep.shape[0]-1)/2),dtype=np.int8)
        count = 0
        for i in xrange(1,keep.shape[0]):
            i_index = keep[i]
            for j in xrange(0,i):
                j_index = keep[j]
                orig_index = i_index*(i_index-1)/2+j_index
                new_ibd[count] = ibd[orig_index]
                count += 1
        return impute_gt(sibgts,new_ibd,f)

### Load pedigree
ped = np.loadtxt(args.ped,dtype='S20',skiprows=1)
#ped = np.loadtxt('relatedness/families.ped',dtype='S20',skiprows=1)

### Create family dictionary
# reverse lookup dict
sib_fam_dict = {}
for i in xrange(0,ped.shape[0]):
    sib_fam_dict[ped[i,1]] = ped[i,0]

### Load IBD
ibd = np.loadtxt(args.ibd,dtype='S20',skiprows=1)
#ibd = np.loadtxt('23andme/ibd_chr_22.txt',dtype='S20',skiprows=1)
# filter by chromosome
if args.king:
    ibd = ibd[ibd[:, 5] == str(chr), :]
else:
    ibd=ibd[ibd[:,2]==str(chr),:]
## split columns
# sibpairs
if args.king:
    ibd_sibs = ibd[:,[1,3]]
else:
    ibd_sibs = ibd[:,0:2]
# map to families
ibd_sibs_in_ped = np.logical_and(np.array([x in sib_fam_dict for x in ibd_sibs[:,0]]),
                                 np.array([x in sib_fam_dict for x in ibd_sibs[:,1]]))
ibd = ibd[ibd_sibs_in_ped,:]
ibd_sibs = ibd_sibs[ibd_sibs_in_ped,:]
ibd_fams = np.array([sib_fam_dict[x] for x in ibd_sibs[:,0]])

# start and end
if args.king:
    bim = args.genotypes.split('.')[0]+'.bim'
    bim = np.loadtxt(bim,dtype='S20')
    # Map SNP IDs to positions
    pos_dict = {}
    for i in range(0,bim.shape[0]):
        pos_dict[bim[i,1]] = int(bim[i,3])
    # Get start and end positions of IBD segments
    ibd_lims = np.zeros((ibd.shape[0],2),dtype=int)
    for i in range(0,ibd.shape[0]):
        ibd_lims[i,0] = pos_dict[ibd[i,8]]
        ibd_lims[i,1] = pos_dict[ibd[i,9]]
else:
    ibd_lims = np.array(ibd[:,np.array([3,6])],dtype=int)

# IBD_2
ibd_2 = np.zeros((ibd.shape[0]),dtype=bool)
if args.king:
    ibd_2[ibd[:,4]=='IBD2'] = True
else:
    ibd_2[ibd[:,4]=='True'] = True
# delete
del ibd

#### Load genotypes
gts_f = Bed(args.genotypes)
#gts_f = Bed('genotypes/chr_22.bed')
gts_ids = gts_f.iid
# Build dict
id_dict = {}
for i in xrange(0,gts_ids.shape[0]):
    id_dict[gts_ids[i,1]] = i

### Identify siblings without genotyped parents
# Remove individuals with genotyped parents
parent_genotyped = np.array([ped[i,2] in id_dict or ped[i,3] in id_dict for i in range(0,ped.shape[0])])
ped = ped[np.logical_not(parent_genotyped),:]
ped_fams = np.unique(ped[:,0])
sibships = {}
sibship_indices = []
for f in ped_fams:
    pedf = ped[ped[:,0]==f,:]
    parent_pairs = np.array([pedf[x, 2] + pedf[x, 3] for x in range(0, pedf.shape[0])])
    unique_parent_pairs = np.unique(parent_pairs)
    pcount = 0
    for par in unique_parent_pairs:
        pmatch = parent_pairs == par
        if np.sum(pmatch)>1:
            sibs = pedf[pmatch, 1]
            sibs_genotyped = np.array([x in id_dict for x in sibs])
            if np.sum(sibs_genotyped)>1:
                sibships[f] = sibs
                sibship_indices = sibship_indices+[id_dict[x] for x in sibs[sibs_genotyped]]
            pcount += 1
    if pcount>1:
        print('More than one sibship without genotyped parents in family '+str(f)+'. Implies incorrect/unsupported pedigree.')

sibship_indices = np.sort(np.unique(np.array(sibship_indices)))

# Read sibling genotypes
print('Reading genotypes')
if args.end is not None:
    gts = gts_f[:, args.start:args.end].read().val
    gts = gts[sibship_indices, :]
    pos = gts_f.pos[args.start:args.end, 2]
    sid = gts_f.sid[args.start:args.end]
else:
    gts = gts_f[sibship_indices, :].read().val
    pos = gts_f.pos[:, 2]
    sid = gts_f.sid

gts = ma.array(gts,mask=np.isnan(gts),dtype=int)

# rebuild ID dictionary
gts_ids = gts_ids[sibship_indices,:]
# Build dict
id_dict = {}
for i in xrange(0,gts_ids.shape[0]):
    id_dict[gts_ids[i,1]] = i

# Calculate allele frequencies
freqs = ma.mean(gts,axis=0)/2.0

nfam = len(sibships)
print('Imputing parental genotypes for '+str(nfam)+' families')
fams = np.sort(np.array(sibships.keys()))
# Impute parental genotypes
imputed_par_gts = np.zeros((nfam,gts.shape[1]),dtype=np.float32)

for i in range(0,nfam):
    # Get sibs in fam
    sibs_i = sibships[fams[i]]
    n_i = len(sibs_i)
    npair_i = n_i*(n_i-1)/2
    # sib dict for family
    sib_dict_i = {}
    for j in xrange(0,n_i):
        sib_dict_i[sibs_i[j]] = j
    # Get sib indices and genotypess
    sib_indices = np.array([id_dict[sibs_i[x]] for x in xrange(0,len(sibs_i))])
    sib_gts = gts[sib_indices, :]
    # Find IBD segments
    ibd_i = np.zeros((npair_i, gts.shape[1]), dtype=np.int8)
    ibd_segs_i = ibd_fams==fams[i]
    nsegs_i = np.sum(ibd_segs_i)
    # Set IBD 1 and 2 segments
    if nsegs_i > 0:
        ibd_lims_i = ibd_lims[ibd_segs_i,:]
        ibd_sibs_i = ibd_sibs[ibd_segs_i,:]
        ibd_2_i = ibd_2[ibd_segs_i]
        for j in xrange(0,nsegs_i):
            pair_indices = [sib_dict_i[x] for x in ibd_sibs_i[j]]
            pindex = pair_index(pair_indices[0],pair_indices[1])
            ibd_i[pindex,np.logical_and(pos>=ibd_lims_i[j,0],pos<=ibd_lims_i[j,1])] += (1+int(ibd_2_i[j]))
    # Impute GTs
    for j in range(0,gts.shape[1]):
        # Deal with missing sib genotypes
        if np.sum(sib_gts[:,j].mask)>0:
            par_impute = 0
            if sib_gts.mask[0, j]:
                par_impute += 2 * freqs[j]
            else:
                par_impute += sib_gts[ 0, j]
            if sib_gts.mask[1, j]:
                par_impute += 2 * freqs[j]
            else:
                par_impute += sib_gts[1, j]
            imputed_par_gts[i, j] = par_impute
        # Or impute based on IBD and observed sib genotypes
        else:
            imputed_par_gts[i, j] = impute_gt(sib_gts[:, j],ibd_i[:,j], freqs[j])
    if np.mod(i, 1000) == 0:
        print('Done '+str(i)+' families')

par_gt_f = h5py.File(args.out+'.hdf5','w')
par_gt_f.create_dataset('imputed_par_gts',imputed_par_gts.shape,dtype = 'f',chunks = True, compression = 'gzip', compression_opts=9)
par_gt_f['imputed_par_gts'][:] = imputed_par_gts
par_gt_f['families'] = fams
par_gt_f['pos'] = pos
par_gt_f['sid'] = sid
par_gt_f.close()
