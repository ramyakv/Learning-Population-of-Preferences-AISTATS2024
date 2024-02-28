import numpy as np
from ranky.metric import kendall_tau_distance
from kemeny import KemenyRanking
from collections import defaultdict, Counter
from itertools import combinations, product
from multiprocessing import Pool


def find_children(key, closure, collector):
    if key not in closure:
        return
    for item in closure[key]:
        collector.add(item)
        find_children(item, closure, collector)


def tc(V):
    # a transitive closure of V
    # the key is greater than the values
    closure = defaultdict(set)

    # add stuff in V first
    for v in V:
        left, right = v.split(" > ")
        closure[left].add(right)

    # add transitive stuff
    for _, value in closure.items():
        collector = set()
        for item in value:
            find_children(item, closure, collector)
        value |= collector

    return closure


def AMP(sigma, closure, phi, m):
    ranking = []
    for i in range(m):
        L = [i_prime for i_prime in range(i) if sigma[i] in closure[ranking[i_prime]]]
        H = [i_prime for i_prime in range(i) if ranking[i_prime] in closure[sigma[i]]]

        l = 0 if len(L) == 0 else max(L) + 1
        h = i if len(H) == 0 else min(H)

        j_candidates = np.arange(l, h + 1)
        j_probs = np.power(phi, i - j_candidates)

        j_probs /= j_probs.sum()
        j = np.random.choice(j_candidates, p=j_probs)
        ranking.insert(j, sigma[i])

    return np.array(ranking)


def dist(closure, sigma):
    d = 0
    for left, right in combinations(sigma, 2):
        d += 1 if left in closure[right] else 0
    return d


def dist_complete(r, sigma):
    return kendall_tau_distance(r, sigma)
    # return kendalltau(r, sigma).statistic
    d = 0
    for left, right in zip(combinations(sigma, 2), combinations(r, 2)):
        d += 1 if left != right else 0
    return d

    
def generate_preferences_from_rankings(rankings, alpha):
    V = []
    for r in rankings:
        v = []
        for x, y in combinations(r, 2):
            if np.random.uniform() > alpha:
                continue
            v.append(str(x) + " > " + str(y))
        V.append(v)
    return V


def generate_preferences(pi_s, sigma_s, n, alpha):
    V = []
    for _ in range(n):
        sigma = sigma_s[np.random.multinomial(1, pi_s).argmax()]
        v = []
        if type(alpha) is float:
            for x, y in combinations(sigma, 2):
                if np.random.uniform() > alpha:
                    continue
                v.append(str(x) + " > " + str(y))
        elif type(alpha) is int:
            candidates = np.array(list(combinations(sigma, 2)))
            v = list(map(lambda x: f'{x[0]} > {x[1]}', candidates[np.random.choice(np.arange(len(candidates)), size=alpha, replace=False)]))
        else:
            raise Exception("alpha must be either float or int")
        V.append(v)
    return V
        

def sample(pi_s, sigma_s, phi_s, m, A, V, T, seed=42):
    np.random.seed(seed)
    K = len(pi_s)

    rankings = []

    for v in V:
        r = topo_sort(v, A)
        for _ in range(T):
            p = np.array([pi_s[k] * np.power(phi_s[k], dist_complete(r, sigma_s[k])) for k in range(K)])
            p = p / p.sum()
            z = np.argmax(np.random.multinomial(1, p))
            r = AMP(sigma=sigma_s[z], closure=tc(v), phi=phi_s[z], m=m)
        rankings.append(r)

    return rankings


def kemeny(S_k, A):
    kr = KemenyRanking(S_k, verbose=False, condorcet_red=False)
    kr.build_Q()
    kr.solve_ilp()
    kr.postprocess()

    return kr.final_solution, kr.obj_sol # / (len(S_k) * (len(S_k) - 1) / 2)


def old_local_kemeny(S_k, sigma_k, A):
    total = 0
    m = len(A)

    c = defaultdict(lambda: defaultdict(int))
    for x, y in product(A, A):
        if x == y:
            continue
        for p in S_k:
            total += 1
            if x not in p or y not in p:
                continue
            if np.where(p == y)[0] < np.where(p == x)[0]:
                c[x][y] += 1

    d = 0
    for x, y in combinations(sigma_k, 2):
        d += c[x][y]
    
    for i in range(1, m):
        x = sigma_k[i]
        for j in reversed(range(i)):
            y = sigma_k[j]
            if c[x][y] < c[y][x]:
                sigma_k[i], sigma_k[j] = sigma_k[j], sigma_k[i]
                d = d + c[x][y] - c[y][x]
            else:
                break

    return sigma_k, d


def topo_sort(v, A):
    A = np.random.permutation(A)
    L = []
    closure = tc(v) # key: from, value: to
    r_closure = defaultdict(set)
    for k, dest in closure.items():
        for item in dest:
            r_closure[item].add(k)

    S = set()
    S.update(r_closure.keys())
    S = list(set(A) - S)
    np.random.shuffle(S)

    while len(S) > 0:
        n = S.pop()
        L.append(n)

        for m in closure[n]:
            r_closure[m].remove(n)
            if len(r_closure[m]) == 0:
                S.append(m)

    return L


def init_param(V, K, A):
    from sklearn.cluster import SpectralClustering, KMeans
    X = np.array(list(map(lambda x: topo_sort(x, A), V)))

    # d_mat = np.zeros((len(X), len(X)))
    # for i, j in combinations(range(len(X)), 2):
    #     d_mat[i, j] = dist_complete(X[i], X[j])
    #     d_mat[j, i] = d_mat[i, j]

    kmeans = KMeans(K, n_init='auto').fit(X)
    # kmeans = SpectralClustering(K, affinity='precomputed_nearest_neighbors').fit(d_mat)

    centorids = []
    d_s = []
    for cluster_idx in range(K):
        centroid, d = kemeny(X[kmeans.labels_ == cluster_idx], A)
        # centroid, d, total = old_local_kemeny(X[kmeans.labels_ == cluster_idx], X[kmeans.labels_ == cluster_idx][np.random.randint(sum(kmeans.labels_ == cluster_idx))], A)
        centorids.append(centroid)
        d_s.append(d)

    p = np.unique(kmeans.labels_, return_counts=True)[1] / len(kmeans.labels_)
    d_s = np.array(d_s) + 1e-12

    return centorids, p, d_s / d_s.sum()

    
def subroutine(args):
    v, A, T, pi_s, phi_s, sigma_s, K, m = args
    r = topo_sort(v, A)
    S = defaultdict(list)
    for _ in range(T):
        p = np.array(pi_s * np.power(phi_s, [dist_complete(r, sigma_s[k]) for k in range(K)])) + 1e-8
        p = p / p.sum()
        z = np.random.choice(a=np.arange(K), p=p)
        r = AMP(sigma=sigma_s[z], closure=tc(v), phi=phi_s[z], m=m)
        S[z].append(r)
    return S
    

def em(V, K, A, T, steps, lr, epoch, seed, verbose=False):
    np.random.seed(seed)

    n = len(V)
    m = len(A)
    # initialize parameters
    sigma_s, pi_s, phi_s = init_param(V, K, A)

    for step in range(steps):
        S = defaultdict(list)
        # E-step
        if len(V) >= 10000:
            with Pool() as p:
                for _set in p.imap_unordered(subroutine, [(v, A, T, pi_s, phi_s, sigma_s, K, m) for v in V]):
                    for k, v in _set.items():
                        S[k] += v
        else:
            for arg in [(v, A, T, pi_s, phi_s, sigma_s, K, m) for v in V]:
                _set = subroutine(arg)
                for k, v in _set.items():
                    S[k] += v

        # M-step
        # optimizing pi
        pi_s = [len(S[k]) / (n*T) for k in range(K)]
        # optimizing sigma
        print(S)
        kemeny_results = [kemeny(S[k], A) for k in range(K)]

        if step > 0 and all([np.all(kemeny_results[k][0] == sigma_s[k]) for k in range(K)]):
            break

        sigma_s = [kemeny_results[k][0] for k in range(K)]
        # optimizing phi
        for _ in range(epoch):
            for k in range(K):
                part_1 = kemeny_results[k][1] / phi_s[k] 
                part_2 = len(S[k]) * sum(
                    [
                        (((i - 1) * phi_s[k] - i) * phi_s[k] ** (i - 1) + 1)
                        / ((1 - phi_s[k] ** i) * (1 - phi_s[k]))
                        for i in range(1, m + 1)
                    ]
                )
                d_phi = part_1 - part_2
                phi_s[k] += lr * d_phi

    return {"pi": pi_s, "sigma": sigma_s, "phi": phi_s}


def main():
    np.random.seed(43)
    K = 3  # number of components
    m = 3  # number of items
    n = 140 * K  # number of agents
    alpha = 1

    A = [str(i) for i in range(m)]  # a set of items
    pi_s = np.random.dirichlet(np.full(K, 5))
    sigma_s = [np.random.permutation(A) for _ in range(K)]
    phi_s = np.random.uniform(0.2, 0.8, size=K)

    V = generate_preferences(pi_s, sigma_s, n, alpha)
    rankings = sample(pi_s, sigma_s, phi_s, m, A, V, T=10, seed=10)
    V = generate_preferences_from_rankings(rankings, alpha)

    print("true pi:")
    for k in range(K):
        print(f"{k}: {pi_s[k]}")

    print("true sigma:")
    for k in range(K):
        print(f"{k}: {sigma_s[k]}")

    print("true phi:")
    for k in range(K):
        print(f"{k}: {phi_s[k]:.5f}")

    result = em(V, K, A, T=10, steps=10, lr=0, epoch=1, seed=1)

    print("inferred pi:")
    for k in range(K):
        print(f"{k}: {result['pi'][k]}")

    print("inferred sigma:")
    for k in range(K):
        print(f"{k}: {result['sigma'][k]}")

    print("inferred phi:")
    for k in range(K):
        print(f"{k}: {result['phi'][k]:.5f}")
    print()


if __name__ == "__main__":
    main()
