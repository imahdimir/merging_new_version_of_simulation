from bgen_reader import open_bgen
import numpy as np
from numba import njit, prange, set_num_threads
from numba import config as numba_config
from sibreg.sibreg import *
import gzip, h5py, os
import snipar.preprocess as preprocess
import snipar.ibd as ibd
from snipar.utilities import *
import argparse

@njit
def impute_from_sibs(g1, g2, ibd, f):
    if ibd==2:
        return g1+2*f
    elif ibd==0:
        return g1+g2
    elif ibd==1:
        gsum = g1+g2
        if gsum==0:
            return f
        elif gsum==1:
            return 1+f
        elif gsum==2:
            return 1+2*f
        elif gsum==3:
            return 2+f
        elif gsum==4:
            return 3+f
@njit
def impute_from_sibs_phased(g1,g2,ibd,f):
    imp = 0.0
    if ibd[0]:
        imp += int(g1[0])+f
    else:
        imp += int(g1[0])+int(g2[0])
    if ibd[1]:
        imp += int(g1[1]) + f
    else:
        imp += int(g1[1]) + int(g2[1])
    return imp

@njit(parallel=True)
def impute_all_fams(gts,freqs,ibd):
    imp = np.zeros((gts.shape[0],gts.shape[2]),dtype=np.float_)
    for i in prange(gts.shape[0]):
        for j in range(gts.shape[2]):
            imp[i,j] = impute_from_sibs(gts[i,0,j],gts[i,1,j],ibd[i,j],freqs[j])
    return imp

@njit(parallel=True)
def impute_all_fams_phased(haps,freqs,ibd):
    imp = np.zeros((haps.shape[0],haps.shape[2]),dtype=np.float_)
    for i in prange(haps.shape[0]):
        for j in range(haps.shape[2]):
            imp[i,j] = impute_from_sibs_phased(haps[i,0,j,:],haps[i,1,j,:],ibd[i,j,:],freqs[j])
    return imp

@njit
def simulate_recombinations(map):
    map_start = map[0]
    map_end = map[map.shape[0]-1]
    map_length = map_end-map_start
    n_recomb = np.random.poisson(map_length/100)
    recomb_points = map_start+np.sort(np.random.uniform(0,1,n_recomb))*map_length
    return n_recomb,recomb_points

@njit
def meiosis(map,n=1):
    # Recomb vector
    recomb_vector = np.zeros((n,map.shape[0]), dtype=np.bool_)
    # Do recombinations
    for r in range(n):
        n_recomb, recomb_points = simulate_recombinations(map)
        # Initialize
        last_hap = np.bool_(np.random.binomial(1,0.5,1)[0])
        recomb_vector[r,:] = np.bool_(last_hap)
        # Recombine
        if n_recomb>0:
            for i in range(n_recomb):
                first_snp = np.argmax(map>recomb_points[i])
                recomb_vector[r,first_snp:recomb_vector.shape[1]] = ~recomb_vector[r,first_snp:recomb_vector.shape[1]]
    # Return
    return recomb_vector


@njit(parallel=True)
def produce_next_gen(father_indices,mother_indices,males,females,map):
    ngen = np.zeros((father_indices.shape[0],2,males.shape[1],2),dtype=np.bool_)
    ibd = np.zeros((father_indices.shape[0],males.shape[1],2),dtype=np.bool_)
    for i in prange(father_indices.shape[0]):
        # recombinations
        recomb_i = meiosis(map,n=4)
        # sib haplotypes and ibd
        for j in range(ibd.shape[1]):
            # paternal sib 1
            if recomb_i[0,j]:
                ngen[i, 0, j, 0] =  males[father_indices[i], j, 0]
            else:
                ngen[i, 0, j, 0] = males[father_indices[i], j, 1]
            # paternal sib 2
            if recomb_i[1,j]:
                ngen[i, 1, j, 0] =  males[father_indices[i], j, 0]
            else:
                ngen[i, 1, j, 0] = males[father_indices[i], j, 1]
            # maternal sib 1
            if recomb_i[2,j]:
                ngen[i, 0, j, 1] =  females[mother_indices[i], j, 0]
            else:
                ngen[i, 0, j, 1] = females[mother_indices[i], j, 1]
            # maternal sib 2
            if recomb_i[3,j]:
                ngen[i, 1, j, 1] =  females[mother_indices[i], j, 0]
            else:
                ngen[i, 1, j, 1] = females[mother_indices[i], j, 1]
            ibd[i,j,0] = recomb_i[0,j]==recomb_i[1,j]
            ibd[i,j,1] = recomb_i[2,j]==recomb_i[3,j]
    return ngen, ibd

@njit
def random_mating_indices(nfam):
    return np.random.choice(np.array([x for x in range(nfam)],dtype=np.int_),size=nfam,replace=False)

def am_indices(yp,ym,r):
    v = np.sqrt(np.var(yp)*np.var(ym))
    s2 = (1/r-1)*v
    zp = yp+np.sqrt(s2)*np.random.randn(yp.shape[0])
    zm = ym+np.sqrt(s2)*np.random.randn(ym.shape[0])
    rank_p = np.argsort(zp)
    rank_m = np.argsort(zm)
    return rank_p, rank_m

def compute_genetic_component(haps,causal,a):
    causal = set(causal)
    G_m = np.zeros((haps[0].shape[0]))
    G_p = np.zeros((haps[0].shape[0]))
    snp_count = 0
    for chr in range(len(haps)):
        snp_index = np.array([snp_count+x for x in range(haps[chr].shape[2])])
        in_causal = np.array([snp_index[x] in causal for x in range(haps[chr].shape[2])])
        causal_gts = np.sum(haps[chr][:,:,in_causal,:],axis=3)
        G_p += causal_gts[:, 0, :].dot(a[snp_index[in_causal]])
        G_m += causal_gts[:, 1, :].dot(a[snp_index[in_causal]])
        snp_count += haps[chr].shape[2]
    return G_p, G_m

def compute_phenotype(haps,causal,a,sigma2):
    G_p, G_m = compute_genetic_component(haps,causal,a)
    Y_p = G_p+np.sqrt(sigma2)*np.random.randn(G_p.shape[0])
    Y_m = G_m+np.sqrt(sigma2)*np.random.randn(G_m.shape[0])
    return G_p, G_m, Y_p, Y_m

def compute_phenotype_vert(haps,causal,a,sigma2,beta_vert,Y_p,Y_m):
    G_males, G_females = compute_genetic_component(haps, causal, a)
    Y_males = G_males + beta_vert*(Y_p+Y_m)+np.sqrt(sigma2) * np.random.randn(G_males.shape[0])
    Y_females = G_females + beta_vert*(Y_p+Y_m)+np.sqrt(sigma2) * np.random.randn(G_females.shape[0])
    return G_males, G_females, Y_males, Y_females

def compute_phenotype_indirect(haps,old_haps,father_indices,mother_indices,causal,a,b,sigma2):
    G_males, G_females = compute_genetic_component(haps,causal,a)
    G_p, G_m = compute_genetic_component(old_haps, causal, b)
    Y_males = G_males+G_p[father_indices]+G_m[mother_indices]+np.sqrt(sigma2)*np.random.randn(G_males.shape[0])
    Y_females = G_females+G_p[father_indices]+G_m[mother_indices]+np.sqrt(sigma2)*np.random.randn(G_males.shape[0])
    return G_males, G_females, Y_males, Y_females

parser = argparse.ArgumentParser()
parser.add_argument('bgenfiles', type=str,
                    help='Address of genotype files in .bgen format (without .bgen suffix). If there is a ~ in the address, ~ is replaced by the chromosome numbers in the range of 1-22.',
                    default=None)
parser.add_argument('h2_direct',type=float,help='Heritability due to direct effects in first generation',default=None)
parser.add_argument('outprefix',type=str,help='Prefix for simulation output files')
parser.add_argument('--n_random',type=int,help='Number of generations of random mating',default=1)
parser.add_argument('--n_am',type=int,help='Number of generations of assortative mating',default=0)
parser.add_argument('--r_par',type=float,help='Phenotypic correlation of parents (for assortative mating)',default=None)
parser.add_argument('--n_causal',type=int,help='Number of causal loci',default=None)
parser.add_argument('--beta_vert',type=float,help='Vertical transmission coefficient',default=0)
parser.add_argument('--h2_total',type=float,help='Total variance explained by direct effects and indirect genetic effects from parents',default=None)
parser.add_argument('--r_dir_indir',type=float,help='Correlation between direct and indirect genetic effects',default=None)
args=parser.parse_args()

if args.beta_vert > 0 and args.h2_total is not None:
    raise(ValueError('Cannot simulate both indirect effects and vertical transmission separately. Choose one'))

if args.ngen_random >= 0:
    ngen_random = args.ngen_random
else:
    raise(ValueError('Number of generations cannot be negative'))

if args.n_am >= 0:
    ngen_am = args.n_am
else:
    raise(ValueError('Number of generations cannot be negative'))

if (args.r_par**2) <= 1:
    r_y = args.r_par
else:
    raise(ValueError('Parental correlation must be between -1 and 1'))

if 0 <= args.h2_direct <= 1:
    h2 = args.h2_direct
else:
    raise(ValueError('Heritability must be between 0 and 1'))

if args.h2_total is not None:
    if args.r_dir_indir is None:
        raise(ValueError('Must specify correlation between direct and indirect genetic effects'))
    else:
        if (args.r_dir_indir**2) <= 1:
            r_dir_alpha = args.r_dir_indir
        else:
            raise(ValueError('Correlation between direct and indirect effects must be between -1 and 1'))
    if 0 <= args.h2_total <= 1:
        h2_total = args.h2_total
    else:
        raise(ValueError('Heritability must be between 0 and 1'))


beta_vert = args.beta_vert
ncausal = args.n_causal

bgenfiles, chroms = preprocess.parse_obsfiles(args.bgenfiles, obsformat='bgen')

# Read genotypes
haps = []
maps = []
snp_ids = []
alleles = []
positions = []
for i in range(bgenfiles.shape[0]):
    print('Reading in chromosome '+str(chroms[i]))
    # Read map
    positions.append(bgen.positions)
    map = preprocess.decode_map_from_pos(chroms[i], bgen.positions)
    not_nan = np.logical_not(np.isnan(map))
    maps.append(map[not_nan])
    # Read bgen
    bgen = open_bgen(bgenfiles[i], verbose=True)
    # Snp
    snp_ids.append(bgen.ids[not_nan])
    # Alleles
    alleles.append(np.array([x.split(',') for x in bgen.allele_ids[not_nan]]))
    # Read genotypes
    gts = bgen.read(([x for x in range(bgen.samples.shape[0])], [x for x in range(bgen.ids.shape[0])]), np.bool_)[:, :,
          np.array([0, 2])]
    gts = gts[:,not_nan]
    nfam = int(np.floor(gts.shape[0] / 2))
    ngen = np.zeros((nfam, 2, gts.shape[1], 2), dtype=np.bool_)
    ngen[:, 0, :, :] = gts[0:nfam, :, :]
    ngen[:, 1, :, :] = gts[nfam:(2 * nfam), :, :]
    del gts
    haps.append(ngen)

# Simulate population
total_matings = ngen_random+ngen_am
V = np.zeros((total_matings+1,2))
a_count = 0
# Produce next generation
old_haps = haps
for gen in range(0, total_matings):
    # Simulate phenotype for AM
    if gen == 0 and h2_total == 0:
        print('Simulating phenotype')
        # Simulate phenotype
        nsnp_chr = np.array([x.shape[2] for x in old_haps])
        nsnp = np.sum(nsnp_chr)
        if ncausal > nsnp:
            raise(ValueError('Not enough SNPs to simulate phenotype with '+str(ncausal)+' causal SNPs'))
        a = np.zeros((nsnp))
        causal = np.sort(np.random.choice(np.arange(0,nsnp),ncausal,replace=False))
        a[causal] = np.random.randn(ncausal)
        G_p, G_m = compute_genetic_component(old_haps,causal,a)
        scale_fctr = np.sqrt(h2/np.var(np.hstack((G_p,G_m))))
        a = a*scale_fctr
    # Compute parental phenotypes
    if np.abs(beta_vert) > 0 and gen>0:
        print('Computing parental phenotypes')
        G_p, G_m, Y_p, Y_m = compute_phenotype_vert(old_haps, causal, a, 1-h2, beta_vert, Y_p[father_indices], Y_m[mother_indices])
    elif h2_total==0:
        print('Computing parental phenotypes')
        G_p, G_m, Y_p, Y_m = compute_phenotype(old_haps, causal, a, 1 - h2)
    # Record variance components
    if gen>0 or h2_total==0:
        V[a_count, :] = np.array([np.var(np.hstack((G_p, G_m))), np.var(np.hstack((Y_p, Y_m)))])
    print('Genetic variance: ' + str(round(V[a_count, 0], 4)))
    print('Phenotypic variance: ' + str(round(V[a_count, 1], 4)))
    print('Heritability: ' + str(round(V[a_count, 0] / V[a_count, 1], 4)))
    a_count += 1
    ## Match parents
    print('Mating ' + str(gen + 1))
    # Random mating
    if gen<ngen_random:
        father_indices = random_mating_indices(nfam)
        mother_indices = random_mating_indices(nfam)
    # Assortative mating
    if gen>=ngen_random:
        # Compute parental phenotypes
        print('Computing parental phenotypes')
        # Match assortatively
        print('Matching assortatively')
        father_indices, mother_indices = am_indices(Y_p, Y_m, 0.5)
        # Print variances
        print('Parental phenotypic correlation: '+str(round(np.corrcoef(Y_p[father_indices],Y_m[mother_indices])[0,1],4)))
        print('Parental genotypic correlation: '+str(round(np.corrcoef(G_p[father_indices],G_m[mother_indices])[0,1],4)))
    # Generate haplotpyes of new generation
    new_haps = []
    ibd = []
    for chr in range(0,len(haps)):
        print('Chromosome '+str(chroms[0]+chr))
        new_haps_chr, ibd_chr = produce_next_gen(father_indices,mother_indices,old_haps[chr][:,0,:,:],old_haps[chr][:,1,:,:],maps[chr])
        new_haps.append(new_haps_chr)
        ibd.append(ibd_chr)
    # Compute indirect effect component
    if h2_total>0:
        if gen==0:
            print('Simulating indirect genetic effects')
            nsnp_chr = np.array([x.shape[2] for x in old_haps])
            nsnp = np.sum(nsnp_chr)
            if ncausal > nsnp:
                raise (ValueError('Not enough SNPs to simulate phenotype with ' + str(ncausal) + ' causal SNPs'))
            ab = np.zeros((nsnp,2))
            causal = np.sort(np.random.choice(np.arange(0, nsnp), ncausal, replace=False))
            ab[causal,:] = np.random.multivariate_normal(np.zeros((2)),
                                                       np.array([[1,r_dir_alpha],[r_dir_alpha,1]]),
                                                       size=ncausal)
            G_males, G_females, Y_males, Y_females = compute_phenotype_indirect(new_haps,old_haps,father_indices,mother_indices,causal,ab[:,0],ab[:,1],0)
            scale_fctr = np.sqrt(h2_total / np.var(np.hstack((Y_males, Y_females))))
            ab = ab*scale_fctr
        print('Computing parental phenotype')
        G_p, G_m, Y_p, Y_m = compute_phenotype_indirect(new_haps,old_haps,father_indices,mother_indices,causal,ab[:,0],ab[:,1],1-h2_total)
    if gen<(total_matings-1):
        old_haps = new_haps

print('Computing final generation phenotypes')
if np.abs(beta_vert)>0:
    G_males, G_females, Y_males, Y_females = compute_phenotype_vert(new_haps, causal, a, 1 - h2, beta_vert, Y_p[father_indices], Y_m[mother_indices])
elif h2_total>0:
    G_males, G_females, Y_males, Y_females = compute_phenotype_indirect(new_haps, old_haps, father_indices, mother_indices, causal,
                                                    ab[:, 0], ab[:, 1], 1 - h2_total)
else:
    G_males, G_females, Y_males, Y_females = compute_phenotype(new_haps, causal, a, 1 - h2)
print('Sibling genotypic correlation: ' + str(round(np.corrcoef(G_males, G_females)[0, 1], 4)))
print('Sibling phenotypic correlation: ' + str(round(np.corrcoef(Y_males, Y_females)[0, 1], 4)))
# Final offspring generation
V[a_count,:] = np.array([np.var(np.hstack((G_males,G_females))),np.var(np.hstack((Y_males, Y_females)))])

print(a)

print('Saving output to file')
# Save variance
vcf = args.outprefix+'VCs.txt'
np.savetxt(vcf, V)
print('Variance components saved to '+str(vcf))
print('Saving pedigree')
# Produce pedigree
ped = np.zeros((nfam*2,6),dtype='U30')
for i in range(0,nfam):
    ped[(2*i):(2*(i+1)),0] = i
    ped[(2 * i):(2 * (i + 1)), 1] = np.array([str(i)+'_0',str(i)+'_1'],dtype=str)
    ped[(2 * i):(2 * (i + 1)),2] = 'P_'+str(i)
    ped[(2 * i):(2 * (i + 1)), 3] = 'M_' + str(i)
    ped[(2 * i):(2 * (i + 1)), 4] = np.array(['0','1'])
    ped[(2 * i):(2 * (i + 1)), 5] = np.array([Y_males[i],Y_females[i]],dtype=ped.dtype)

sibpairs = ped[:,1].reshape((int(ped.shape[0]/2),2))
ped = np.vstack((np.array(['FID','IID','FATHER_ID','MOTHER_ID','SEX','PHENO']),ped))
np.savetxt(args.outprefix+'sibs.ped',ped[:,0:4],fmt='%s')
np.savetxt(args.outprefix+'sibs.fam',ped[1:ped.shape[0],:],fmt='%s')

## Save to HDF5 file
print('Saving genotypes to '+args.outprefix+'genotypes.hdf5')
hf = h5py.File(args.outprefix+'genotypes.hdf5','w')
# save pedigree
hf['ped'] = encode_str_array(ped)
# save offspring genotypes
chr_count = 0
for i in range(len(haps)):
    print('Writing genotypes for chromosome '+str(chroms[i]))
    bim_i = np.hstack((snp_ids[i], maps[i], positions[i], alleles[i]))
    hf['chr_'+str(chroms[i])+'_bim'] = encode_str_array(bim_i)
    gts_chr = np.sum(new_haps[i], axis=3, dtype=np.uint8)
    hf['chr_'+str(chroms[i])] = gts_chr.reshape((gts_chr.shape[0]*2,gts_chr.shape[2]),order='C')
    # # Imputed parental genotypes
    # print('Imputing parental genotypes and saving')
    # freqs = np.mean(gts_chr, axis=(0, 1)) / 2.0
    # # Phased
    # phased_imp = impute_all_fams_phased(new_haps[chr_count],freqs,ibd[chr_count])
    # hf['phased_imp_chr'+str(chr)] = phased_imp
    # # Unphased
    # ibd[chr_count] = np.sum(ibd[chr_count],axis=2)
    # imp = impute_all_fams(gts_chr, freqs, ibd[chr_count])
    # hf['imp_chr_'+str(chr)] = imp
    # Parental genotypes
    # print('Saving true parental genotypes')
    # par_gts_chr = np.zeros((old_haps[chr_count].shape[0],2,old_haps[chr_count].shape[2]),dtype=np.uint8)
    # par_gts_chr[:,0,:] = np.sum(old_haps[chr_count][father_indices,0,:,:],axis=2)
    # par_gts_chr[:,1,:] = np.sum(old_haps[chr_count][mother_indices,1,:,:],axis=2)
    # hf['par_chr_'+str(chr)] = par_gts_chr
    chr_count += 1

hf.close()

# Write IBD segments
snp_count = 0

if h2_total>0:
    causal_out = np.zeros((ab.shape[0], 8), dtype='U30')
else:
    causal_out = np.zeros((a.shape[0],7),dtype='U30')
for i in range(len(haps)):
    print('Writing IBD segments for chromosome '+str(chroms[i]))
    # Segments
    segs = ibd.write_segs_from_matrix(ibd[chr_count], sibpairs,
                                  snp_ids[i], positions[i],maps[i],chroms[i],
                                  args.outprefix+'chr_'+str(chr)+'.segments.gz')
    # Causal effects
    if h2_total==0:
        a_chr = a[snp_count:(snp_count + snp_ids[i].shape[0])]
        causal_out[snp_count:(snp_count + snp_ids[i].shape[0]),:] = np.hstack((snp_ids[i],alleles[i],a_chr.reshape((snp_ids[i].shape[0],1))))
    else:
        ab_chr = ab[snp_count:(snp_count + snp_ids[i].shape[0]),:]
        causal_out[snp_count:(snp_count + snp_ids[i].shape[0]),:] = np.hstack((snp_ids[i],alleles[i],ab_chr.reshape((snp_ids[i].shape[0],2))))
    snp_count += snp_ids[i].shape[0]
    # count
    chr_count += 1

np.savetxt(args.outprefix+'causal_effects.txt',causal_out,fmt='%s')