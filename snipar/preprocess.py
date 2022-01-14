from os import path
from snipar.gtarray import gtarray
from pysnptools.snpreader import Bed
import numpy.ma as ma
from numba import njit, prange
import numpy as np
from snipar.utilities import make_id_dict

def parse_obsfiles(obsfiles, obsformat='bed'):
    obs_files = []
    if '~' in obsfiles:
        bed_ixes = obsfiles.split('~')
        for i in range(1,23):
            obsfile = bed_ixes[0]+str(i)+bed_ixes[1]+'.'+obsformat
            if path.exists(obsfile):
                obs_files.append(obsfile)
        print(str(len(obs_files))+' observed genotype files found')
    else:
            obs_files = [obsfiles+'.'+obsformat]
    return np.array(obs_files)

def parse_filelist(obsfiles, impfiles, obsformat):
    obs_files = []
    imp_files = []
    if '~' in obsfiles and impfiles:
        bed_ixes = obsfiles.split('~')
        imp_ixes = impfiles.split('~')
        for i in range(1,23):
            obsfile = bed_ixes[0]+str(i)+bed_ixes[1]+'.'+obsformat
            impfile = imp_ixes[0]+str(i)+imp_ixes[1]+'.hdf5'
            if path.exists(impfile) and path.exists(obsfile):
                obs_files.append(obsfile)
                imp_files.append(impfile)
        print(str(len(imp_files))+' matched observed and imputed genotype files found')
    else:
            obs_files = [obsfiles+'.'+obsformat]
            imp_files = [impfiles+'.hdf5']
    return np.array(obs_files), np.array(imp_files)

def get_sibpairs_from_ped(ped):
    parent_missing = np.array([ped[i,2]=='0' or ped[i,3]=='0' for i in range(ped.shape[0])])
    #print('Removing '+str(np.sum(parent_missing))+' rows from pedigree due to missing parent(s)')
    ped = ped[np.logical_not(parent_missing),:]
    # Find unique parent-pairs
    parent_pairs = np.array([ped[i,2]+ped[i,3] for i in range(ped.shape[0])])
    unique_pairs, sib_counts = np.unique(parent_pairs, return_counts=True)
    # Parent pairs with more than one offspring
    sib_parent_pairs = unique_pairs[sib_counts>1]
    fam_sizes = np.array([x*(x-1)/2 for x in sib_counts[sib_counts>1]])
    npairs = int(np.sum(fam_sizes))
    if npairs==0:
        raise(ValueError('No sibling pairs found'))
    # Find all sibling pairs
    sibpairs = np.zeros((npairs,2),dtype=ped.dtype)
    paircount = 0
    for ppair in sib_parent_pairs:
        sib_indices = np.where(parent_pairs==ppair)[0]
        for i in range(0,sib_indices.shape[0]-1):
            for j in range(i+1,sib_indices.shape[0]):
                sibpairs[paircount,:] = np.array([ped[sib_indices[i],1],ped[sib_indices[j],1]])
                paircount += 1
    return sibpairs

def find_individuals_with_sibs(ids, ped, gts_ids, return_ids_only = False):
    """
    Used in get_gts_matrix and get_fam_means to find the individuals in ids that have genotyped siblings.
    """
    # Find genotyped sibships of size > 1
    ped_dict = make_id_dict(ped, 1)
    ids_in_ped = np.array([x in ped_dict for x in gts_ids])
    gts_fams = np.zeros((gts_ids.shape[0]),dtype=gts_ids.dtype)
    gts_fams[ids_in_ped] = np.array([ped[ped_dict[x], 0] for x in gts_ids[ids_in_ped]])
    fams, counts = np.unique(gts_fams[ids_in_ped], return_counts=True)
    sibships = set(fams[counts > 1])
    # Find individuals with genotyped siblings
    ids_in_ped = np.array([x in ped_dict for x in ids])
    ids = ids[ids_in_ped]
    ids_fams = np.array([ped[ped_dict[x], 0] for x in ids])
    ids_with_sibs = np.array([x in sibships for x in ids_fams])
    ids = ids[ids_with_sibs]
    ids_fams = ids_fams[ids_with_sibs]
    if return_ids_only:
        return ids
    else:
        return ids, ids_fams, gts_fams

def get_sibpairs_from_king(kinfile):
    kin_header = np.array(open(kinfile,'r').readline().split('\t'))
    inf_type_index = np.where(np.array([x[0:7]=='InfType' for x in kin_header]))[0][0]
    id1_index = np.where(np.array(kin_header)=='ID1')[0][0]
    id2_index = np.where(np.array(kin_header)=='ID2')[0][0]
    sibpairs = np.loadtxt(kinfile,dtype=str,skiprows=1,usecols=(id1_index,id2_index,inf_type_index))
    sibpairs = sibpairs[sibpairs[:,2]=='FS',0:2]
    return sibpairs

def get_indices_given_ped(ped, fams, gts_ids, ids=None, sib=False, verbose = False):
    """
    Used in get_gts_matrix_given_ped to get the ids of individuals with observed/imputed parental genotypes and, if sib=True, at least one genotyped sibling.
    It returns those ids along with the indices of the relevant individuals and their first degree relatives in the observed genotypes (observed indices),
    and the indices of the imputed parental genotypes for those individuals.
    """
    # Made dictionary for observed genotypes
    gts_id_dict = make_id_dict(gts_ids)
    # If IDs not provided, use all individuals with observed genotypes
    if ids is None:
        ids = gts_ids
    # Find individuals with siblings
    if sib:
        ids = find_individuals_with_sibs(ids, ped, gts_ids, return_ids_only=True)
        if verbose:
            print('Found ' + str(ids.shape[0]) + ' individuals with genotyped siblings')
    ### Find parental status
    if verbose:
        print('Checking for observed/imputed parental genotypes')
    par_status, gt_indices, fam_labels = find_par_gts(ids, ped, fams, gts_id_dict)
    # Find which individuals can be used
    none_missing = np.min(gt_indices, axis=1)
    none_missing = none_missing >= 0
    N = np.sum(none_missing)
    if N == 0:
        raise ValueError(
            'No individuals with phenotype observations and complete observed/imputed genotype observations')
    if verbose:
        print(str(N) + ' individuals with phenotype observations and complete observed/imputed genotypes observations')
    # Take those that can be used
    gt_indices = gt_indices[none_missing, :]
    par_status = par_status[none_missing, :]
    ids = ids[none_missing]
    # Find indices of individuals and their parents in observed genotypes
    observed_indices = np.sort(np.unique(np.hstack((gt_indices[:, 0],
                                                    gt_indices[par_status[:, 0] == 0, 1],
                                                    gt_indices[par_status[:, 1] == 0, 2]))))
    # Get indices of imputed parents
    imp_indices = np.sort(np.unique(np.hstack((gt_indices[par_status[:, 0] == 1, 1],
                                               gt_indices[par_status[:, 1] == 1, 2]))))
    # Return ids with imputed/observed parents
    return ids, observed_indices, imp_indices

def get_indices_given_ped(ped, fams, gts_ids, ids=None, sib=False, verbose = False):
    """
    Used in get_gts_matrix_given_ped to get the ids of individuals with observed/imputed parental genotypes and, if sib=True, at least one genotyped sibling.
    It returns those ids along with the indices of the relevant individuals and their first degree relatives in the observed genotypes (observed indices),
    and the indices of the imputed parental genotypes for those individuals.
    """
    # Made dictionary for observed genotypes
    gts_id_dict = make_id_dict(gts_ids)
    # If IDs not provided, use all individuals with observed genotypes
    if ids is None:
        ids = gts_ids
    # Find individuals with siblings
    if sib:
        ids = find_individuals_with_sibs(ids, ped, gts_ids, return_ids_only=True)
        if verbose:
            print('Found ' + str(ids.shape[0]) + ' individuals with genotyped siblings')
    ### Find parental status
    if verbose:
        print('Checking for observed/imputed parental genotypes')
    par_status, gt_indices, fam_labels = find_par_gts(ids, ped, fams, gts_id_dict)
    # Find which individuals can be used
    none_missing = np.min(gt_indices, axis=1)
    none_missing = none_missing >= 0
    N = np.sum(none_missing)
    if N == 0:
        raise ValueError(
            'No individuals with phenotype observations and complete observed/imputed genotype observations')
    if verbose:
        print(str(N) + ' individuals with phenotype observations and complete observed/imputed genotypes observations')
    # Take those that can be used
    gt_indices = gt_indices[none_missing, :]
    par_status = par_status[none_missing, :]
    ids = ids[none_missing]
    # Find indices of individuals and their parents in observed genotypes
    observed_indices = np.sort(np.unique(np.hstack((gt_indices[:, 0],
                                                    gt_indices[par_status[:, 0] == 0, 1],
                                                    gt_indices[par_status[:, 1] == 0, 2]))))
    # Get indices of imputed parents
    imp_indices = np.sort(np.unique(np.hstack((gt_indices[par_status[:, 0] == 1, 1],
                                               gt_indices[par_status[:, 1] == 1, 2]))))
    # Return ids with imputed/observed parents
    return ids, observed_indices, imp_indices

def estimate_genotyping_error_rate(bedfiles,ped):
    ME_het = np.zeros((2))
    for bedfile in bedfiles:
        ME_het += mendelian_errors_from_bed(bedfile,ped)
    p = ME_het[0]/ME_het[1]
    return p

def mendelian_errors_from_bed(bedfile,ped):
    # Read bed
    bed = Bed(bedfile, count_A1=True)
    ids = bed.iid
    id_dict = make_id_dict(ids, 1)
    ## Find parent-offspring pairs
    # genotyped individuals
    genotyped = np.array([x in id_dict for x in ped[:, 1]])
    ped = ped[genotyped, :]
    # with genotyped father
    father_genotyped = np.array([x in id_dict for x in ped[:, 2]])
    # with genotyped mother
    mother_genotyped = np.array([x in id_dict for x in ped[:, 3]])
    # either
    opg = np.logical_or(father_genotyped, mother_genotyped)
    opg_ped = ped[opg, :]
    # number of pairs
    npair = np.sum(father_genotyped) + np.sum(mother_genotyped)
    if npair == 0:
        raise(ValueError('No parent-offspring pairs in  '+str(bedfile)+' for genotype error probability estimation'))
    print(str(npair)+' parent-offspring pairs found in '+bedfile)
    if npair*bed.sid.shape[0] < 10**5:
        print('Warning: limited information for estimation of genotyping error probability.')
    ## Read genotypes
    all_ids = np.unique(np.hstack((opg_ped[:, 1],
                                   ped[father_genotyped, 2],
                                   ped[mother_genotyped, 3])))
    all_ids_indices = np.sort(np.array([id_dict[x] for x in all_ids]))
    gts = bed[all_ids_indices, :].read().val
    id_dict = make_id_dict(ids[all_ids_indices, :], 1)
    ## Get indices
    pair_indices = np.zeros((npair,2),dtype=int)
    pair_count = 0
    for i in range(opg_ped.shape[0]):
        o_index = id_dict[opg_ped[i,1]]
        if opg_ped[i, 2] in id_dict:
            pair_indices[pair_count,:] = np.array([o_index,id_dict[opg_ped[i,2]]])
            pair_count += 1
        if opg_ped[i, 3] in id_dict:
            pair_indices[pair_count,:] = np.array([o_index,id_dict[opg_ped[i,3]]])
            pair_count += 1
    ## Count Mendelian errors
    ME = count_ME(gts,pair_indices)
    # Compute allele frequencies
    gts = ma.array(gts, mask=np.isnan(gts))
    freqs = ma.mean(gts, axis=0) / 2.0
    # Estimate error probability
    sum_het = np.sum(freqs * (1 - freqs))
    return np.array([ME,sum_het*pair_count])

@njit(parallel=True)
def count_ME(gts,pair_indices):
    ME = 0
    # Count Mendelian errors
    for i in prange(pair_indices.shape[0]):
        for j in range(gts.shape[1]):
                if np.abs(gts[pair_indices[i,0],j]-gts[pair_indices[i,1],j])>1:
                    ME += 1
    return ME

# Read header of mapfile
def get_map_positions(mapfile,gts,min_map_prop = 0.5):
    map_file = open(mapfile,'r')
    map_header = map_file.readline()
    map_header = np.array(map_header.split(' '))
    map_header[len(map_header)-1] = map_header[len(map_header)-1].split('\n')[0]
    map_file.close()
    if 'pposition' in map_header and 'gposition' in map_header:
        bp_pos = np.loadtxt(mapfile,usecols = np.where(map_header=='pposition')[0][0], dtype=int, skiprows =1)
        pos_dict = make_id_dict(bp_pos)
        cm_pos = np.loadtxt(mapfile,usecols = np.where(map_header=='gposition')[0][0], dtype=float, skiprows =1)
        # Check for NAs
        if np.sum(np.isnan(cm_pos)) > 0:
            raise (ValueError('Map cannot have NAs'))
        if np.min(cm_pos) < 0:
            raise (ValueError('Map file cannot have negative values'))
        if np.var(cm_pos) == 0:
            raise (ValueError('Map file has no variation'))
        # Check ordering
        ordered_map = np.sort(cm_pos)
        if np.array_equal(cm_pos, ordered_map):
            pass
        else:
            raise (ValueError('Map not monotonic. Please make sure input is ordered correctly'))
        # Check scale
        if np.max(cm_pos) > 5000:
            raise (ValueError('Maximum value of map too large'))
        # Find positions of SNPs in map file
        map = np.zeros((gts.shape[1]),dtype=float)
        map[:] = np.nan
        in_map = np.array([x in pos_dict for x in gts.pos])
        # Check if we have at least 50% of SNPs in map
        prop_in_map = np.mean(in_map)
        if prop_in_map < min_map_prop:
            raise(ValueError('Only '+str(round(100*prop_in_map))+'% of SNPs have genetic positions in '+mapfile+'. Need at least '+str(round(100*min_map_prop))+'%'))
        print('Found genetic map positions for '+str(round(100*prop_in_map))+'% of SNPs in '+mapfile)
        # Fill in map values
        map[in_map] = cm_pos[[pos_dict[x] for x in gts.pos[in_map]]]
        # Linearly interpolate map
        if prop_in_map < 1:
            print('Linearly interpolating genetic map for SNPs not in input map')
            map = np.interp(gts.pos, gts.pos[in_map], map[in_map])
        return map
    else:
        raise(ValueError('Map file must contain columns pposition and gposition'))

#### Compute LD-scores ####
@njit(parallel=True)
def compute_ld_scores(gts,map,max_dist = 1):
    ldscores = np.zeros((gts.shape[1]),dtype=np.float64)
    for i in prange(gts.shape[1]):
        ldscore_i = 1
        if i>0:
            j = i-1
            dist = map[i]-map[j]
            while dist < max_dist and j>=0:
                ldscore_i += r2_est(gts[...,i],gts[...,j])
                j -= 1
                if j>=0:
                    dist = map[i]-map[j]
        if i<(gts.shape[1]-1):
            j = i + 1
            dist = map[j] - map[i]
            while dist < max_dist and j < gts.shape[1]:
                ldscore_i += r2_est(gts[..., i], gts[..., j])
                j += 1
                if j < gts.shape[1]:
                    dist = map[j] - map[i]
        ldscores[i] = ldscore_i
    return ldscores

## Unbiased estimator of R^2 between SNPs
@njit
def r2_est(g1,g2):
    not_nan = np.logical_not(np.logical_or(np.isnan(g1), np.isnan(g2)))
    r2 = np.power(np.corrcoef(g1[not_nan],g2[not_nan])[0,1],2)
    return r2-(1-r2)/(np.sum(not_nan)-2)

def get_fam_means(ids,ped,gts,gts_ids,remove_proband = True, return_famsizes = False):
    """
    Used in get_gts_matrix to find the mean genotype in each sibship (family) for each SNP or for a PGS.
    The gtarray that is returned is indexed based on the subset of ids provided from sibships of size 2 or greater.
    If remove_proband=True, then the genotype/PGS of the index individual is removed from the fam_mean given for that individual.
    """
    ids, ids_fams, gts_fams = find_individuals_with_sibs(ids,ped,gts_ids)
    fams = np.unique(ids_fams)
    fams_dict = make_id_dict(fams)
    # Compute sums of genotypes in each family
    fam_sums = np.zeros((fams.shape[0],gts.shape[1]),dtype=gts.dtype)
    fam_counts = np.zeros((fams.shape[0]),dtype=int)
    for i in range(0,fams.shape[0]):
        fam_indices = np.where(gts_fams==fams[i])[0]
        fam_sums[i,:] = np.sum(gts[fam_indices,:],axis=0)
        fam_counts[i] = fam_indices.shape[0]
    # Place in vector corresponding to IDs
    if remove_proband:
        gts_id_dict = make_id_dict(gts_ids)
    G_sib = np.zeros((ids.shape[0],gts.shape[1]),dtype = np.float32)
    for i in range(0,ids.shape[0]):
        fam_index = fams_dict[ids_fams[i]]
        G_sib[i,:] = fam_sums[fam_index,:]
        n_i = fam_counts[fam_index]
        if remove_proband:
            G_sib[i,:] = G_sib[i,:] - gts[gts_id_dict[ids[i]],:]
            n_i = n_i-1
        G_sib[i,:] = G_sib[i,:]/float(n_i)
    if return_famsizes:
        return [gtarray(G_sib, ids),fam_counts,fam_sums]
    else:
        return gtarray(G_sib,ids)

def find_par_gts(pheno_ids, ped, fams, gts_id_dict):
    """
    Used in get_gts_matrix to find whether individuals have imputed or observed parental genotypes, and to
    find the indices of the observed/imputed parents in the observed/imputed genotype arrays.
    'par_status' codes whether an individual has parents that are observed or imputed or neither.
    'gt_indices' records the relevant index of the parent in the observed/imputed genotype arrays
    'fam_labels' records the family of the individual based on the pedigree
    """
    # Whether mother and father have observed/imputed genotypes
    par_status = np.zeros((pheno_ids.shape[0],2),dtype=int)
    par_status[:] = -1
    # Indices of obsered/imputed genotypes in relevant arrays
    gt_indices = np.zeros((pheno_ids.shape[0],3),dtype=int)
    gt_indices[:] = -1
    ## Build dictionaries
    # Where each individual is in the pedigree
    ped_dict = make_id_dict(ped,1)
    # Where the imputed data is for each family
    fam_dict = make_id_dict(fams)
    # Store family ID of each individual
    fam_labels = np.zeros((pheno_ids.shape[0]),dtype=fams.dtype)
    # Find status and find indices
    for i in range(0,pheno_ids.shape[0]):
        # Find index in genotypes
        if pheno_ids[i] in gts_id_dict:
            gt_indices[i,0] = gts_id_dict[pheno_ids[i]]
        # Find index in pedigree
        if pheno_ids[i] in ped_dict:
            ped_i = ped[ped_dict[pheno_ids[i]], :]
            fam_labels[i] = ped_i[0]
            # Check for observed father
            if ped_i[2] in gts_id_dict:
                gt_indices[i,1] = gts_id_dict[ped_i[2]]
                par_status[i,0] = 0
            # Check for observed mother
            if ped_i[3] in gts_id_dict:
                gt_indices[i, 2] = gts_id_dict[ped_i[3]]
                par_status[i,1] = 0
            # If parent not observed, look for imputation
            if ped_i[0] in fam_dict:
                imp_index = fam_dict[ped_i[0]]
                # Check if this is imputation of father, or mother, or both
                if ped_i[4] == 'False' and not par_status[i,0] == 0:
                    gt_indices[i, 1] = imp_index
                    par_status[i, 0] = 1
                if ped_i[5] == 'False' and not par_status[i,1] == 0:
                    gt_indices[i, 2] = imp_index
                    par_status[i, 1] = 1
    return par_status, gt_indices, fam_labels

def make_gts_matrix(gts,imp_gts,par_status,gt_indices, parsum = False):
    """
    Used in get_gts_matrix to construct the family based genotype matrix given
    observed/imputed genotypes. 'gt_indices' has the indices in the observed/imputed genotype arrays;
    and par_status codes whether the parents are observed (0) or imputed (1).
    """
    if np.min(gt_indices)<0:
        raise(ValueError('Missing genotype index'))
    N = gt_indices.shape[0]
    if parsum:
        gdim = 2
    else:
        gdim = 3
    G = np.zeros((N,gdim,gts.shape[1]),np.float32)
    # Proband genotypes
    G[:,0,:] = gts[gt_indices[:,0],:]
    # Paternal genotypes
    G[par_status[:, 0] == 0, 1 ,:] = gts[gt_indices[par_status[:, 0] == 0, 1], :]
    G[par_status[:, 0] == 1, 1, :] = imp_gts[gt_indices[par_status[:, 0] == 1, 1], :]
    # Maternal genotypes
    if parsum:
        G[par_status[:, 1] == 0, 1, :] += gts[gt_indices[par_status[:, 1] == 0, 2], :]
        G[par_status[:, 1] == 1, 1, :] += imp_gts[gt_indices[par_status[:, 1] == 1, 2], :]
    else:
        G[par_status[:, 1] == 0, 2, :] = gts[gt_indices[par_status[:, 1] == 0, 2], :]
        G[par_status[:, 1] == 1, 2, :] = imp_gts[gt_indices[par_status[:, 1] == 1, 2], :]
    return G
