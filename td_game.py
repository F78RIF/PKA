import pygame, sys, math, heapq, random, time
GRID_SIZE_DEFAULT = 20
CELL_DEFAULT = 32
FPS = 60
WINDOW_MARGIN = 150
SPAWN_INTERVAL = 1000 
WAVE_SIZE = 8

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid_blocked, start, goal, cost_map=None):
    rows = len(grid_blocked); cols = len(grid_blocked[0])
    openq = []
    heapq.heappush(openq, (manhattan(start, goal), 0, start, None))
    came = {}
    gscore = {start: 0}
    while openq:
        f, g, cur, parent = heapq.heappop(openq)
        if cur in came: continue
        came[cur] = parent
        if cur == goal:
            path = []; node = cur
            while node:
                path.append(node); node = came[node]
            path.reverse(); return path
        x,y = cur
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx, y+dy
            if 0<=nx<cols and 0<=ny<rows and not grid_blocked[ny][nx]:
                step_cost = 1.0
                if cost_map: step_cost = cost_map[ny][nx]
                ng = g + step_cost
                neigh = (nx,ny)
                if ng < gscore.get(neigh, 1e9):
                    gscore[neigh] = ng
                    heapq.heappush(openq, (ng + manhattan(neigh, goal), ng, neigh, cur))
    return None

def dijkstra(grid_blocked, start, goal, cost_map=None):
    rows = len(grid_blocked); cols = len(grid_blocked[0])
    pq = []
    heapq.heappush(pq, (0, start, None))
    dist = {start: 0}
    parent = {}
    while pq:
        d, cur, par = heapq.heappop(pq)
        if cur in parent: continue
        parent[cur] = par
        if cur == goal:
            path = []; node = cur
            while node:
                path.append(node); node = parent[node]
            path.reverse(); return path
        x,y = cur
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx, y+dy
            if 0<=nx<cols and 0<=ny<rows and not grid_blocked[ny][nx]:
                step_cost = 1.0
                if cost_map: step_cost = cost_map[ny][nx]
                nd = d + step_cost
                neigh = (nx,ny)
                if nd < dist.get(neigh, 1e9):
                    dist[neigh] = nd
                    heapq.heappush(pq, (nd, neigh, cur))
    return None

# ---------- Game objects ----------
class Grid:
    def __init__(self, cols=GRID_SIZE_DEFAULT, rows=GRID_SIZE_DEFAULT, cell_px=CELL_DEFAULT):
        self.cols = cols; self.rows = rows; self.cell_px = cell_px
        self.width = cols * cell_px; self.height = rows * cell_px
        self.cells = [[False]*cols for _ in range(rows)]  # True = wall/blocked
        self.start = (0, rows//2); self.goal = (cols-1, rows//2)

    def reset_empty(self):
        self.cells = [[False]*self.cols for _ in range(self.rows)]

    def block(self, cell):
        x,y = cell
        if 0<=x<self.cols and 0<=y<self.rows:
            self.cells[y][x] = True

    def unblock(self, cell):
        x,y = cell
        if 0<=x<self.cols and 0<=y<self.rows:
            self.cells[y][x] = False

    def is_blocked(self, cell):
        x,y = cell
        return self.cells[y][x]

    def pixel_to_cell(self, pos):
        x,y = pos
        return (x // self.cell_px, y // self.cell_px)

    def cell_center(self, cell):
        cx,cy = cell
        return (cx*self.cell_px + self.cell_px//2, cy*self.cell_px + self.cell_px//2)

    def matrix(self):
        return self.cells

class Enemy:
    def __init__(self, path, hp=10, speed=1.0, is_boss=False):
        self.path = path or []
        self.pos = None
        self.path_index = 0
        self.hp = hp; self.max_hp = hp
        self.speed = speed  # cells per second
        self.alive = True
        self.is_boss = is_boss
        self.spawn_time = pygame.time.get_ticks()

    def set_start_pixel(self, grid):
        if self.path:
            cx,cy = grid.cell_center(self.path[0])
            self.pos = (float(cx), float(cy))
            self.path_index = 1 if len(self.path)>1 else 0

    def update(self, dt, grid=None):
        if not self.path or not self.alive: return
        if self.pos is None:
            if grid: self.set_start_pixel(grid)
            else: return
        if self.path_index >= len(self.path):
            self.alive = False; return 'goal'
        tx,ty = grid.cell_center(self.path[self.path_index])
        px,py = self.pos
        vx,vy = tx-px, ty-py
        dist = math.hypot(vx,vy)
        if dist < 1e-4:
            if self.path_index >= len(self.path)-1:
                self.alive = False; return 'goal'
            else:
                self.path_index += 1; return None
        speed_px = self.speed * grid.cell_px
        move = speed_px * dt
        if move >= dist:
            self.pos = (tx,ty)
            if self.path_index >= len(self.path)-1:
                self.alive = False; return 'goal'
            else:
                self.path_index += 1
        else:
            nx = px + (vx/dist)*move
            ny = py + (vy/dist)*move
            self.pos = (nx, ny)
        return None

    def cell(self, grid=None):
        if self.pos and grid:
            cx = int(self.pos[0] // grid.cell_px)
            cy = int(self.pos[1] // grid.cell_px)
            cx = max(0, min(grid.cols-1, cx)); cy = max(0, min(grid.rows-1, cy))
            return (cx, cy)
        if self.path:
            idx = min(self.path_index, len(self.path)-1)
            return self.path[idx]
        return None

class Tower:
    def __init__(self, cell, level=1):
        self.cell = cell; self.level = level
        self.base_range = 2.5; self.base_damage = 3; self.base_fire_rate = 1.0
        self.cooldown = 0.0

    @property
    def range(self): return self.base_range * (1 + 0.4*(self.level-1))
    @property
    def damage(self): return int(self.base_damage * (1 + 0.6*(self.level-1)))
    @property
    def fire_rate(self): return self.base_fire_rate * (1 + 0.5*(self.level-1))

    def update(self, dt, enemies, grid):
        self.cooldown -= dt
        if self.cooldown > 0: return
        tx,ty = self.cell
        target=None; bestd=1e9
        for e in enemies:
            if not e.alive: continue
            ec = e.cell(grid)
            if not ec: continue
            ex,ey = ec
            d = math.hypot(ex-tx, ey-ty)
            if d <= self.range and d < bestd:
                bestd=d; target=e
        if target:
            target.hp -= self.damage
            if target.hp <= 0: target.alive=False
            self.cooldown = 1.0/self.fire_rate

    def upgrade(self):
        if self.level < 3: self.level += 1

# ---------- UI Button ----------
class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect); self.label = label
    def draw(self, surf, font, active=False):
        color = (120,200,120) if active else (160,160,160)
        pygame.draw.rect(surf, color, self.rect); pygame.draw.rect(surf, (30,30,30), self.rect, 2)
        txt = font.render(self.label, True, (0,0,0)); surf.blit(txt, (self.rect.x+6, self.rect.y+6))
    def clicked(self, pos): return self.rect.collidepoint(pos)

# ---------- Game ----------
class Game:
    def __init__(self):
        pygame.init()
        self.grid = Grid(cols=GRID_SIZE_DEFAULT, rows=GRID_SIZE_DEFAULT, cell_px=CELL_DEFAULT)
        self.window = pygame.display.set_mode((self.grid.width, self.grid.height + WINDOW_MARGIN))
        pygame.display.set_caption("TD Game - towers non-blocking")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)

        # game state
        self.towers = []
        self.enemies = []
        self.last_spawn = pygame.time.get_ticks()
        self.spawned = 0
        self.wave = 1
        self.money = 100
        self.lives = 10
        self.algorithm = 'A*'
        self.running = True

        self.algorithm_cost_map = None
        self.wave_start_time = pygame.time.get_ticks()
        self.boss_spawned_this_wave = False

        # UI buttons
        bx = 10; by = self.grid.height + 10
        self.btn_regen = Button((bx, by, 120, 28), 'Regenerate Map')
        self.btn_astar = Button((bx+130, by, 80, 28), 'A*')
        self.btn_dij = Button((bx+220, by, 110, 28), 'Dijkstra')
        self.btn_perf = Button((bx+340, by, 140, 28), 'Perf Test (P)')

        # generate map
        self.generate_random_map(wall_prob=0.18)

    def generate_random_map(self, wall_prob=0.18, max_attempts=120):
        attempts = 0
        cols,rows = self.grid.cols, self.grid.rows
        while attempts < max_attempts:
            attempts += 1
            self.grid.reset_empty()
            sx,sy = random.randrange(cols), random.randrange(rows)
            gx,gy = random.randrange(cols), random.randrange(rows)
            while (gx,gy) == (sx,sy):
                gx,gy = random.randrange(cols), random.randrange(rows)
            self.grid.start = (sx,sy); self.grid.goal = (gx,gy)
            for y in range(rows):
                for x in range(cols):
                    if (x,y) == self.grid.start or (x,y) == self.grid.goal:
                        continue
                    if random.random() < wall_prob:
                        self.grid.block((x,y))
            p = astar(self.grid.matrix(), self.grid.start, self.grid.goal)
            if p:
                self.make_cost_map()
                self.towers = []
                self.enemies = []
                self.spawned = 0
                self.last_spawn = pygame.time.get_ticks()
                self.wave = 1
                self.money = 100
                self.lives = 10
                self.wave_start_time = pygame.time.get_ticks()
                self.boss_spawned_this_wave = False
                return
        self.grid.reset_empty()
        self.grid.start = (0, self.grid.rows//2); self.grid.goal = (self.grid.cols-1, self.grid.rows//2)
        self.make_cost_map()
        self.towers = []; self.enemies = []; self.spawned = 0

    def make_cost_map(self):
        cols,rows = self.grid.cols, self.grid.rows
        mat = [[1.0 for _ in range(cols)] for _ in range(rows)]
        if self.algorithm == 'Dijkstra':
            for y in range(rows):
                for x in range(cols):
                    if (x,y)==self.grid.start or (x,y)==self.grid.goal:
                        mat[y][x] = 1.0
                    elif self.grid.cells[y][x]:
                        mat[y][x] = 9999.0
                    else:
                        mat[y][x] = 1.0 + random.random()*1.2
        else:
            for y in range(rows):
                for x in range(cols):
                    mat[y][x] = 1.0 if not self.grid.cells[y][x] else 9999.0
        self.algorithm_cost_map = mat

    def compute_path(self):
        mat = self.grid.matrix()
        start = self.grid.start; goal = self.grid.goal
        if not self.algorithm_cost_map or len(self.algorithm_cost_map)!=self.grid.rows or len(self.algorithm_cost_map[0])!=self.grid.cols:
            self.make_cost_map()
        if self.algorithm == 'A*':
            return astar(mat, start, goal, cost_map=self.algorithm_cost_map)
        else:
            return dijkstra(mat, start, goal, cost_map=self.algorithm_cost_map)

    def spawn_enemy(self, is_boss=False):
        path = self.compute_path()
        if not path: return
        base_hp = 8 + (self.wave-1)*4
        if is_boss:
            hp = base_hp*6 + 40; speed = 0.55
            e = Enemy(path=path, hp=hp, speed=speed, is_boss=True)
        else:
            hp = base_hp; speed = 1.0 + 0.08*(self.wave-1)
            e = Enemy(path=path, hp=hp, speed=speed)
        e.set_start_pixel(self.grid)
        self.enemies.append(e)

    def reroute_enemies(self):
        new_path = self.compute_path()
        if not new_path: return
        for e in self.enemies:
            if not e.alive: continue
            cur = e.cell(self.grid)
            best_idx, bestd = 0, 1e9
            for i, cell in enumerate(new_path):
                d = abs(cell[0]-cur[0]) + abs(cell[1]-cur[1])
                if d < bestd:
                    bestd = d; best_idx = i
            e.path = new_path
            e.set_start_pixel(self.grid)
            e.path_index = best_idx

    # ---------- towers: now non-blocking ----------
    def place_tower(self, cell):
        # disallow if max towers
        if len(self.towers) >= 3: return False
        # cannot place on start/goal or wall
        if cell == self.grid.start or cell == self.grid.goal: return False
        if self.grid.is_blocked(cell): return False
        # cannot place if tower already exists there
        for t in self.towers:
            if t.cell == cell: return False
        # cost
        cost = 20
        if self.money < cost: return False

        # NOTE: tower placement no longer blocks the grid.
        # We therefore do NOT block/unblock cells nor reroute enemies here.
        # This allows towers to be placed even on the computed path; enemies will still walk.
        self.towers.append(Tower(cell))
        self.money -= cost
        return True

    def upgrade_tower(self, cell):
        for t in self.towers:
            if t.cell == cell:
                cost = 25 * t.level
                if self.money >= cost and t.level < 3:
                    self.money -= cost; t.upgrade(); return True
        return False

    def run_performance_test(self, trials=50, block_ratio=0.18):
        cols,rows = self.grid.cols, self.grid.rows
        start=(0,rows//2); goal=(cols-1,rows//2)
        astar_times=[]; dijk_times=[]; found=0
        for _ in range(trials):
            mat = [[False]*cols for _ in range(rows)]
            for y in range(rows):
                for x in range(cols):
                    if (x,y) in (start,goal): continue
                    mat[y][x] = random.random() < block_ratio
            cm_astar = [[1.0 if not mat[y][x] else 9999.0 for x in range(cols)] for y in range(rows)]
            cm_dij = [[1.0 + random.random()*1.2 if not mat[y][x] else 9999.0 for x in range(cols)] for y in range(rows)]
            t0=time.perf_counter(); p1=astar(mat, start, goal, cost_map=cm_astar); t1=time.perf_counter()
            t2=time.perf_counter(); p2=dijkstra(mat, start, goal, cost_map=cm_dij); t3=time.perf_counter()
            if p1 or p2: found+=1
            astar_times.append((t1-t0)*1000.0); dijk_times.append((t3-t2)*1000.0)
        avg_a=sum(astar_times)/len(astar_times); avg_d=sum(dijk_times)/len(dijk_times)
        self.perf_summary = f"Perf {trials} runs | found ~{found}/{trials} | A* {avg_a:.3f}ms | D {avg_d:.3f}ms"
        print(self.perf_summary)

    # ---------- update & draw ----------
    def update(self, dt):
        now = pygame.time.get_ticks()
        if now - self.last_spawn >= SPAWN_INTERVAL and self.spawned < WAVE_SIZE:
            self.spawn_enemy(); self.spawned += 1; self.last_spawn = now
        if not self.boss_spawned_this_wave and (now - self.wave_start_time) >= 10000:
            self.spawn_enemy(is_boss=True); self.boss_spawned_this_wave = True
        if self.spawned >= WAVE_SIZE and all(not e.alive for e in self.enemies):
            self.wave += 1; self.spawned = 0; self.enemies = []; self.money += 25
            self.wave_start_time = pygame.time.get_ticks(); self.boss_spawned_this_wave = False

        for t in self.towers: t.update(dt, self.enemies, self.grid)

        for e in list(self.enemies):
            if not e.alive:
                if e.hp <= 0: self.money += (12 if e.is_boss else 5)
                else: self.lives -= 1
                if e in self.enemies: self.enemies.remove(e)
                continue
            res = e.update(dt, grid=self.grid)
            if res == 'goal':
                self.lives -= 1; e.alive = False
                if e in self.enemies: self.enemies.remove(e)

    def draw(self):
        win = self.window; win.fill((28,28,28))
        # grid + walls
        for y in range(self.grid.rows):
            for x in range(self.grid.cols):
                rect = pygame.Rect(x*self.grid.cell_px, y*self.grid.cell_px, self.grid.cell_px, self.grid.cell_px)
                if self.grid.is_blocked((x,y)):
                    pygame.draw.rect(win, (50,50,70), rect)   # wall color
                else:
                    pygame.draw.rect(win, (60,60,60), rect)
                pygame.draw.rect(win, (18,18,18), rect, 1)

        # draw computed path highlight
        path = self.compute_path()
        if path:
            for cell in path:
                cx,cy = self.grid.cell_center(cell)
                srect = pygame.Rect(cx - self.grid.cell_px//4, cy - self.grid.cell_px//4, self.grid.cell_px//2, self.grid.cell_px//2)
                pygame.draw.rect(win, (80,120,80), srect)

        # start & goal
        sx,sy = self.grid.start; gx,gy = self.grid.goal
        pygame.draw.rect(win, (40,200,40), (sx*self.grid.cell_px, sy*self.grid.cell_px, self.grid.cell_px, self.grid.cell_px))
        pygame.draw.rect(win, (200,40,40), (gx*self.grid.cell_px, gy*self.grid.cell_px, self.grid.cell_px, self.grid.cell_px))

        # towers (now may overlap path; that's intended)
        for t in self.towers:
            cx,cy = self.grid.cell_center(t.cell)
            pygame.draw.circle(win, (220,200,100), (cx,cy), max(6, self.grid.cell_px//3))
            pygame.draw.circle(win, (220,200,100), (cx,cy), int(t.range * self.grid.cell_px), 1)
            txt = self.font.render(str(t.level), True, (0,0,0)); win.blit(txt, (cx-6, cy-8))

        # enemies
        for e in self.enemies:
            if not e.path: continue
            if getattr(e, 'pos', None):
                px,py = int(e.pos[0]), int(e.pos[1])
            else:
                cell = e.cell(self.grid); px,py = self.grid.cell_center(cell)
            frac = max(0, e.hp) / e.max_hp
            w = int(self.grid.cell_px * 0.8); h = 6
            if e.is_boss:
                radius = max(10, self.grid.cell_px//2)
                pygame.draw.rect(win, (150,30,30), (px - w//2, py - self.grid.cell_px//2 + 6, int(w*frac), h+6))
                pygame.draw.circle(win, (220,80,80), (px,py), radius)
                pygame.draw.circle(win, (180,40,40), (px,py), radius, 3)
            else:
                pygame.draw.rect(win, (200,40,40), (px - w//2, py - self.grid.cell_px//2 + 4, int(w*frac), h))
                pygame.draw.circle(win, (180,180,250), (px,py), max(4, self.grid.cell_px//4))

        # UI
        self.btn_regen.draw(win, self.font)
        self.btn_astar.draw(win, self.font, active=(self.algorithm=='A*'))
        self.btn_dij.draw(win, self.font, active=(self.algorithm=='Dijkstra'))
        self.btn_perf.draw(win, self.font)

        ui_y = self.grid.height + 46
        info = f"Map: {self.grid.cols}x{self.grid.rows} | Alg: {self.algorithm} | Wave: {self.wave} | Money: {self.money} | Lives: {self.lives} | Towers: {len(self.towers)}/3"
        win.blit(self.font.render(info, True, (230,230,230)), (10, ui_y))
        win.blit(self.font.render("Click grid: place tower (20). U: upgrade. A/D: change algo. R: regen map. P: perf test", True, (200,200,200)), (10, ui_y+20))
        if hasattr(self, 'perf_summary'):
            win.blit(self.font.render(getattr(self, 'perf_summary', ''), True, (240,220,140)), (10, ui_y+45))

        pygame.display.flip()

    def run(self):
        while self.running and self.lives > 0:
            dt = self.clock.tick(FPS) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_a:
                        if self.algorithm != 'A*':
                            self.algorithm = 'A*'; self.make_cost_map(); self.reroute_enemies()
                    elif ev.key == pygame.K_d:
                        if self.algorithm != 'Dijkstra':
                            self.algorithm = 'Dijkstra'; self.make_cost_map(); self.reroute_enemies()
                    elif ev.key == pygame.K_u:
                        mx,my = pygame.mouse.get_pos(); cell = self.grid.pixel_to_cell((mx,my)); self.upgrade_tower(cell)
                    elif ev.key == pygame.K_r:
                        self.generate_random_map()
                    elif ev.key == pygame.K_p:
                        self.run_performance_test()
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 1:
                        mx,my = ev.pos
                        if self.btn_regen.clicked((mx,my)):
                            self.generate_random_map(); continue
                        if self.btn_astar.clicked((mx,my)):
                            if self.algorithm != 'A*':
                                self.algorithm='A*'; self.make_cost_map(); self.reroute_enemies(); continue
                        if self.btn_dij.clicked((mx,my)):
                            if self.algorithm != 'Dijkstra':
                                self.algorithm='Dijkstra'; self.make_cost_map(); self.reroute_enemies(); continue
                        if self.btn_perf.clicked((mx,my)):
                            self.run_performance_test(); continue
                        cell = self.grid.pixel_to_cell((mx,my))
                        if 0<=cell[0]<self.grid.cols and 0<=cell[1]<self.grid.rows:
                            placed = self.place_tower(cell)
                            if not placed:
                                self.upgrade_tower(cell)
            self.update(dt)
            self.draw()

        # end
        self.window.fill((10,10,10))
        msg = "Game Over" if self.lives<=0 else "Exited"
        self.window.blit(self.font.render(msg, True, (240,240,240)), (self.grid.width//2 - 40, self.grid.height//2))
        pygame.display.flip()
        pygame.time.wait(1500)
        pygame.quit()

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
