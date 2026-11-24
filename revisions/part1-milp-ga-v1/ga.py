"""Single-objective GA. No access to the MILP solution or optimum."""
import random
from image_quality.optimize import evaluate, feasibility

VERSIONS = ('low', 'medium', 'high')


class Choices:
    def __init__(self, rows, limits):
        self.ids = list(limits)
        self.rows = [
            {
                VERSIONS.index(r['version']): r
                for r in rows
                if r['image_id'] == i and r['ssim'] >= limits[i]
            }
            for i in self.ids
        ]
        self.eligible = [tuple(sorted(group)) for group in self.rows]

    def selected(self, genes):
        return [group[g] for group, g in zip(self.rows, genes)]

    def size(self, genes):
        return sum((group[g]['bytes'] for group, g in zip(self.rows, genes)))

    def score(self, genes):
        return sum((group[g]['ssim'] for group, g in zip(self.rows, genes))) / len(self.ids)

    def smallest(self):
        return tuple((min(group, key=lambda g: (group[g]['bytes'], g)) for group in self.rows))

    def repair(self, genes, budget):
        """Every step strictly decreases integer bytes, including nonmonotonic cases."""
        genes = list(genes)
        size = self.size(genes)
        moves = 0

        while size > budget:
            replacements = []
            for i, (group, g) in enumerate(zip(self.rows, genes)):
                current = group[g]
                for replacement, candidate in group.items():
                    saved = current['bytes'] - candidate['bytes']
                    if saved > 0:
                        loss = current['ssim'] - candidate['ssim']
                        replacements.append((loss / saved, i, replacement, saved))

            if not replacements:
                raise ValueError('No cheaper eligible replacement: budget infeasible')

            _, i, replacement, saved = min(replacements)
            genes[i] = replacement
            size -= saved
            moves += 1

        return (tuple(genes), moves)


def mutate(genes, eligible, probability, rng):
    child = list(genes)

    for i, choices in enumerate(eligible):
        if len(choices) > 1 and rng.random() < probability:
            child[i] = rng.choice([g for g in choices if g != child[i]])

    return tuple(child)


def solve_ga(rows, limits, budget, settings, seed):
    if type(budget) is not int or budget < 0:
        raise ValueError('Budget must be nonnegative integer bytes')

    preflight = feasibility(rows, limits)
    if preflight['images_without_qualifying_candidate'] or budget < preflight['minimum_feasible_bytes']:
        return dict(
            evaluate([], limits, budget),
            budget_bytes=budget,
            status='infeasible_precheck',
            preflight=preflight,
            seed=seed,
            generations_completed=0,
            fitness_evaluations=0,
            history=[],
            optimality_gap=None,
            solver_status_code=None,
            solve_seconds=None
        )

    population_size = settings['population']
    elite = settings['elitism']

    if not 0 <= elite < population_size or population_size < 2:
        raise ValueError('Invalid population or elitism')

    rng = random.Random(seed)
    choices = Choices(rows, limits)
    population = [choices.smallest()]
    repair_calls = repair_moves = 0

    def repair(genes):
        nonlocal repair_calls, repair_moves
        fixed, moves = choices.repair(genes, budget)
        repair_calls += 1
        repair_moves += moves

        return fixed

    while len(population) < population_size:
        population.append(repair(tuple((rng.choice(v) for v in choices.eligible))))

    history = []
    evaluations = 0

    for generation in range(settings['generations'] + 1):
        scored = sorted([(choices.score(g), g) for g in population], key=lambda p: (-p[0], p[1]))
        evaluations += len(population)
        history.append({
            'generation': generation,
            'best_ssim': scored[0][0],
            'mean_population_ssim': sum((p[0] for p in scored)) / population_size,
            'distinct_chromosomes': len(set(population))
        })

        if generation == settings['generations']:
            break

        def tournament():
            return min(
                [rng.choice(scored) for _ in range(settings['tournament'])],
                key=lambda p: (-p[0], p[1])
            )[1]

        following = [g for _, g in scored[:elite]]
        while len(following) < population_size:
            a, b = (tournament(), tournament())
            if rng.random() < settings['crossover_probability']:
                mask = [rng.random() < 0.5 for _ in a]
                children = [
                    tuple((x if keep else y for x, y, keep in zip(a, b, mask))),
                    tuple((y if keep else x for x, y, keep in zip(a, b, mask)))
                ]
            else:
                children = [a, b]

            for child in children:
                if len(following) == population_size:
                    break

                following.append(repair(mutate(
                    child,
                    choices.eligible,
                    settings['mutation_probability'],
                    rng
                )))

        population = following

    best = scored[0][1]
    result = evaluate(choices.selected(best), limits, budget)

    return dict(
        result,
        budget_bytes=budget,
        status='completed_heuristic',
        preflight=preflight,
        seed=seed,
        genes=list(best),
        generations_completed=settings['generations'],
        fitness_evaluations=evaluations,
        repair_calls=repair_calls,
        repair_moves=repair_moves,
        history=history,
        optimality_gap=None,
        solver_status_code=None,
        solve_seconds=None
    )
