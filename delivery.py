# Problème de tournées de drones avec Tabu Search et 2-Opt

from math import sqrt

# -------------------------------
# Données et structures de base
# -------------------------------

class Customer:
    def __init__(self, idx, x, y, demand=0.0):
        self.id = idx
        self.x = x
        self.y = y
        self.demand = demand

def euclid(a: Customer, b: Customer) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return sqrt(dx*dx + dy*dy)

def distance_matrix(customers):
    n = len(customers)
    d = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            val = euclid(customers[i], customers[j])
            d[i][j] = val
            d[j][i] = val
    return d

# -------------------------------
# Évaluation d’une solution
# -------------------------------

class Problem:
    def __init__(self, customers, vehicle_count, capacity, max_route_len,
                 cap_penalty=1000.0, dist_penalty=1000.0):
        self.customers = customers
        self.n = len(customers)
        self.depot = 0
        self.m = vehicle_count
        self.capacity = capacity
        self.max_route_len = max_route_len
        self.cap_penalty = cap_penalty
        self.dist_penalty = dist_penalty
        self.D = distance_matrix(customers)

    def route_distance(self, route):
        # route: [0, ..., 0]
        total = 0.0
        for i in range(len(route)-1):
            total += self.D[route[i]][route[i+1]]
        return total

    def route_load(self, route):
        # sum of demands excluding depot
        load = 0.0
        for c in route:
            if c != self.depot:
                load += self.customers[c].demand
        return load

    def route_penalty(self, route):
        load = self.route_load(route)
        dist = self.route_distance(route)
        over_cap = max(0.0, load - self.capacity)
        over_dist = max(0.0, dist - self.max_route_len)
        return self.cap_penalty * over_cap + self.dist_penalty * over_dist

    def route_cost(self, route):
        return self.route_distance(route) + self.route_penalty(route)

    def solution_cost(self, solution):
        # solution: list of routes
        return sum(self.route_cost(r) for r in solution)

# -------------------------------
# Outils manipulation de routes
# -------------------------------

def clone_solution(sol):
    return [r[:] for r in sol]

def flatten_customers(solution):
    # Liste de tous les clients (sans depot) présents
    res = []
    for r in solution:
        for c in r:
            if c != 0:
                res.append(c)
    return res

def verify_all_customers_present(problem, solution):
    seen = sorted(flatten_customers(solution))
    expected = list(range(1, problem.n))
    return seen == expected

# -------------------------------
# Construction initiale
# -------------------------------

def greedy_seed(problem: Problem):
    # Simple: balayage du plus proche voisin en respectant capacité/autonomie (si possible)
    unserved = set(range(1, problem.n))
    solution = []
    for _ in range(problem.m):
        route = [0]
        load = 0.0
        dist_used = 0.0
        cur = 0
        while unserved:
            best = None
            best_gain = None
            for c in list(unserved):
                d_add = problem.D[cur][c] + problem.D[c][0] - problem.D[cur][0]
                new_load = load + problem.customers[c].demand
                # estimer distance si on insère c puis rentre au dépôt
                new_dist = problem.route_distance(route + [c, 0])
                # admissibilité souple: on permet violations (pénalisées) pour construire
                if best_gain is None or d_add < best_gain:
                    best_gain = d_add
                    best = c
            if best is None:
                break
            # On ajoute best si cela ne rend pas la route grotesque; sinon on clôture
            tentative = route[:-1] + [best, 0] if route[-1] == 0 else route + [best, 0]
            # On accepte même si violation; si trop long (heuristique), on ferme
            if problem.route_distance(tentative) > problem.max_route_len * 1.5:
                break
            route = tentative
            load += problem.customers[best].demand
            dist_used = problem.route_distance(route)
            unserved.remove(best)
            cur = best
        if route == [0]:
            # route vide -> laisser pour plus tard
            continue
        # S'assurer qu'elle se termine par le dépô t
        if route[-1] != 0:
            route.append(0)
        solution.append(route)
        if not unserved:
            break

    # Si des clients restent, créer des routes simples 0-c-0
    for c in sorted(unserved):
        solution.append([0, c, 0])

    # S'il manque des véhicules, remplir avec routes vides 0-0
    while len(solution) < problem.m:
        solution.append([0, 0])

    return solution

# -------------------------------
# 2-Opt intra-tournée
# -------------------------------

def two_opt_improve(problem: Problem, route):
    if len(route) <= 4:
        return route, False
    best = route[:]
    best_cost = problem.route_cost(best)
    improved = False
    # i et k sont les coupures, on ne casse pas les 0 de début/fin
    for i in range(1, len(route)-2):
        for k in range(i+1, len(route)-1):
            # inversion du segment [i:k]
            new_route = route[:i] + list(reversed(route[i:k+1])) + route[k+1:]
            cost = problem.route_cost(new_route)
            if cost + 1e-9 < best_cost:
                best = new_route
                best_cost = cost
                improved = True
    return best, improved

def two_opt_full(problem: Problem, route, max_pass=5):
    current = route[:]
    for _ in range(max_pass):
        current, ok = two_opt_improve(problem, current)
        if not ok:
            break
    return current

def apply_two_opt_all(problem: Problem, solution):
    changed = False
    new_sol = []
    for r in solution:
        new_r = two_opt_full(problem, r)
        if new_r != r:
            changed = True
        new_sol.append(new_r)
    return new_sol, changed

# -------------------------------
# Génération de voisins (Relocate & Swap)
# -------------------------------

def generate_neighbors(problem: Problem, solution, max_neighbors_per_pair=64):
    nsol = []
    # Relocate: déplacer un client c de (ra, ia) vers (rb, posb)
    for ra_idx, ra in enumerate(solution):
        for ia in range(1, len(ra)-1):  # positions clients
            c = ra[ia]
            for rb_idx, rb in enumerate(solution):
                for posb in range(1, len(rb)):  # insertion avant posb
                    if ra_idx == rb_idx and (posb == ia or posb == ia+1):
                        continue
                    new_sol = clone_solution(solution)
                    # retirer c de ra
                    del new_sol[ra_idx][ia]
                    # ajuster pos si même route et posb > ia
                    adj_posb = posb
                    if rb_idx == ra_idx and posb > ia:
                        adj_posb -= 1
                    new_sol[rb_idx].insert(adj_posb, c)
                    nsol.append(("relocate", (c, ra_idx, rb_idx), new_sol))
                    if len(nsol) > max_neighbors_per_pair and rb_idx != ra_idx:
                        break  # limiter un peu
                # fin positions insertion
            # fin boucle rb
        # fin positions ra
    # Swap: échanger deux clients (c1) et (c2)
    S = len(solution)
    for ra_idx in range(S):
        for ia in range(1, len(solution[ra_idx])-1):
            c1 = solution[ra_idx][ia]
            for rb_idx in range(ra_idx, S):
                jb_start = 1
                if rb_idx == ra_idx:
                    jb_start = ia+1
                for jb in range(jb_start, len(solution[rb_idx])-1):
                    c2 = solution[rb_idx][jb]
                    new_sol = clone_solution(solution)
                    new_sol[ra_idx][ia], new_sol[rb_idx][jb] = new_sol[rb_idx][jb], new_sol[ra_idx][ia]
                    nsol.append(("swap", (c1, c2, ra_idx, rb_idx), new_sol))
    return nsol

# -------------------------------
# Gestion Tabu
# -------------------------------

class TabuList:
    def __init__(self, tenure=15):
        self.tenure = tenure
        self.table = {}  # key -> remaining iterations

    def step(self):
        to_del = []
        for k in list(self.table.keys()):
            self.table[k] -= 1
            if self.table[k] <= 0:
                to_del.append(k)
        for k in to_del:
            del self.table[k]

    def forbid(self, key):
        self.table[key] = self.tenure

    def is_tabu(self, key):
        return key in self.table

# clé Tabu simplifiée:
# - relocate: ("r", customer_id, target_route_idx)
# - swap:     ("s", min_customer, max_customer)

def tabu_key(move_type, data):
    if move_type == "relocate":
        c, _ra, rb = data
        return ("r", c, rb)
    elif move_type == "swap":
        c1, c2, _ra, _rb = data
        a, b = (c1, c2) if c1 < c2 else (c2, c1)
        return ("s", a, b)
    return ("x",)

# -------------------------------
# Tabu Search
# -------------------------------

def tabu_search(problem: Problem, init_solution,
                max_iter=400, tenure=20, patience=100):
    # Appliquer 2-Opt initial
    current = clone_solution(init_solution)
    current, _ = apply_two_opt_all(problem, current)
    best = clone_solution(current)
    best_cost = problem.solution_cost(best)

    tabu = TabuList(tenure=tenure)
    no_improve = 0

    for it in range(1, max_iter+1):
        neighbors = generate_neighbors(problem, current)
        best_candidate = None
        best_candidate_cost = float("inf")
        best_candidate_move = None
        # Évaluer voisins
        for (mtype, mdata, sol) in neighbors:
            # Amélioration locale 2-Opt rapide par route impactée
            # Pour efficacité, on pourrait ne 2-opt que les routes modifiées, mais ici simplicité:
            sol2, _ = apply_two_opt_all(problem, sol)
            cost = problem.solution_cost(sol2)

            key = tabu_key(mtype, mdata)
            tabu_blocked = tabu.is_tabu(key)

            # Aspiration: si améliore le global, on ignore tabu
            if tabu_blocked and cost + 1e-9 >= best_cost:
                continue

            if cost + 1e-9 < best_candidate_cost:
                best_candidate = sol2
                best_candidate_cost = cost
                best_candidate_move = (mtype, mdata, key)

        # Si aucun voisin admis, on relâche la tabu list (diversification)
        if best_candidate is None:
            tabu.step()
            no_improve += 1
            if no_improve > patience:
                break
            continue

        # Accepter le meilleur voisin
        current = best_candidate
        tabu.forbid(best_candidate_move[2])
        tabu.step()

        # Mettre à jour le meilleur global
        if best_candidate_cost + 1e-9 < best_cost:
            best = clone_solution(current)
            best_cost = best_candidate_cost
            no_improve = 0
        else:
            no_improve += 1

        # Option: arrêt anticipé si stabilité
        if no_improve > patience:
            break

    return best, best_cost

# -------------------------------
# Affichage et vérifications
# -------------------------------

def print_solution(problem, solution):
    total_dist = 0.0
    total_pen = 0.0
    print("Solution:")
    for idx, r in enumerate(solution):
        dist = problem.route_distance(r)
        pen = problem.route_penalty(r)
        load = problem.route_load(r)
        total_dist += dist
        total_pen += pen
        print(f"  Drone {idx+1}: route={r} | dist={dist:.3f} | load={load:.2f} | penalty={pen:.3f}")
    print(f"Total distance: {total_dist:.3f}")
    print(f"Total penalties: {total_pen:.3f}")
    print(f"Objective (dist+pen): {total_dist + total_pen:.3f}")

# -------------------------------
# Exemple d’utilisation
# -------------------------------

def demo():
    # Dépôt + 15 clients
    customers = [
        Customer(0, 0.0, 0.0, 0.0),   # depot
        Customer(1, 10.0, 5.0, 0.8),
        Customer(2, 12.0, -3.0, 0.7),
        Customer(3, -4.0, 7.0, 1.2),
        Customer(4, -6.0, -2.0, 0.9),
        Customer(5, 3.0, 9.0, 0.6),
        Customer(6, 8.0, 10.0, 1.0),
        Customer(7, 15.0, 2.0, 0.5),
        Customer(8, 9.0, -8.0, 0.9),
        Customer(9, -10.0, 1.0, 1.1),
        Customer(10, -8.0, 9.0, 0.7),
        Customer(11, 6.0, -12.0, 0.6),
        Customer(12, 2.0, 14.0, 0.8),
        Customer(13, 14.0, -6.0, 0.9),
        Customer(14, -12.0, -5.0, 1.0),
        Customer(15, -2.0, -10.0, 0.8),
    ]

    drones = 5
    capacity = 3.0            # kg
    max_route_len = 50.0      # autonomie (distance max par tournée)

    problem = Problem(customers, drones, capacity, max_route_len,
                      cap_penalty=1500.0, dist_penalty=1200.0)

    init_sol = greedy_seed(problem)
    # S’assurer que tous les clients sont servis
    assert verify_all_customers_present(problem, init_sol), "Clients manquants dans la solution initiale"

    print("Solution initiale (greedy + 2-Opt):")
    init_sol, _ = apply_two_opt_all(problem, init_sol)
    print_solution(problem, init_sol)

    best_sol, best_cost = tabu_search(problem, init_sol,
                                      max_iter=300,
                                      tenure=25,
                                      patience=60)

    print("\nSolution finale (Tabu Search + 2-Opt):")
    print_solution(problem, best_sol)


demo()
