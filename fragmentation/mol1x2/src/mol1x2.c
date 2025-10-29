#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/* ========= 修复：定义M_PI常量 ========= */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========= 全局常量（与 Fortran 版本一致的上限） ========= */
#define MAX_ATOMS      1997
#define MAX_STATES     1997000   /* 3 * MAX_STATES 作为坐标池 */
#define DEG2RAD(d)     ((d) * (float)M_PI / 180.0f)
#define RAD2DEG(r)     ((r) * 180.0f / (float)M_PI)
#define MIN(a,b)       ((a) < (b) ? (a) : (b))
#define MAX(a,b)       ((a) > (b) ? (a) : (b))
#define ENERGY_THRESHOLD 0.1f
/* ========= 修复：统一定义能量转换因子 ========= */
#define KCAL_TO_AU 0.001593601f  /* 1 kcal/mol = 0.001593601 a.u. */
#define EV_TO_AU   0.036749309f  /* 1 eV = 0.036749309 a.u. */
#define AU_TO_KCAL 627.509474f   /* 1 a.u. = 627.509474 kcal/mol */
#define AU_TO_EV   27.21138602f  /* 1 a.u. = 27.21138602 eV */


/* ========= 全局数据（Fortran: glob_module） ========= */
static float glob_thresh_e0 = 0.16f, glob_thresh_e1 = 0.16f, glob_thresh_e2 = 0.16f, glob_thresh_e3 = 0.16f;

/* 原子“库仑半径”/“范德华半径”近似表（与 Fortran 保持一致，后续未定义元素填 1.6） */
static float glob_atom_radi[121] = {
/* 1..38:  Fortran 的 data 段（到 Sr） */
  0.38f,0.55f,1.54f,1.13f,0.80f,0.71f,0.69f,0.67f,0.63f,0.65f,1.90f,1.60f,1.43f,1.17f,1.04f,1.02f,0.99f,1.00f,2.35f,1.97f,1.61f,1.54f,1.31f,1.25f,1.81f,1.25f,1.25f,1.24f,1.28f,1.33f,1.23f,1.22f,1.16f,1.15f,1.14f,0.99f,2.43f,2.15f,
/* 39..120：C 里补齐，默认 1.6（与 Fortran main 里循环一致） */
};
static char glob_ele_table[121][3]; /* 两字符元素符号 + 终止符 */
static int  glob_ele_table_inited = 0;

/* Fortran 里的一串元素字符串，这里变成数组初始化 */
static const char *ELE_STR =
" H He Li Be  B  C  N  O  F Ne Na Mg Al Si  P  S Cl Ar  K Ca Sc Ti  V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr  Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te  I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta  W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn";

/* ========= 小工具：元素符号 ? 原子序号 ========= */
static void init_element_table(void) {
    if (glob_ele_table_inited) return;
    /* 先把 1..38 的半径表处理好了，随后 39..120 填 1.6 */
    for (int i = 38; i < 120; ++i) {
        glob_atom_radi[i] = 1.6f;
    }
    /* 解析 ELE_STR，每个元素占 2 字符（可能有空格前缀），Fortran 用 3 列宽；这里按空白切分更稳健 */
    int idx = 1;
    const char *p = ELE_STR;
    while (*p && idx <= 120) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char buf[8] = {0};
        int bi = 0;
        while (*p && !isspace((unsigned char)*p) && bi < 2) buf[bi++] = *p++;
        buf[bi] = '\0';
        strncpy(glob_ele_table[idx], buf, 3);
        idx++;
        while (*p && !isspace((unsigned char)*p)) p++;
    }
    glob_ele_table_inited = 1;
}

static int elem_symbol_to_Z(const char *sym) {
    init_element_table();
    char t[3] = {0};
    /* 统一大小写：首字母大写、次字母小写 */
    int len = (int)strlen(sym);
    if (len == 0) return 0;
    t[0] = (char)toupper((unsigned char)sym[0]);
    if (len > 1) t[1] = (char)tolower((unsigned char)sym[1]);
    for (int i = 1; i <= 120; ++i) {
        if (strcmp(glob_ele_table[i], t) == 0) return i;
    }
    return 0;
}

/* ========= 读能量标题（Fortran: sf_get_energy_from_title） ========= */
static int get_energy_from_title(const char *title_line, float *energy_out) {
    /* 按 Fortran 的多个关键字顺序尝试匹配，允许 "Energy:" "energy:" "Energy" "e:" 等 */
    const char *keys[] = {"Energy:", "energy:", "Energy", "energy", "TStmpE:", "E:", "e:"};
    for (int k = 0; k < 7; ++k) {
        const char *pos = strstr(title_line, keys[k]);
        if (!pos) continue;
        pos += strlen(keys[k]);
        while (*pos && isspace((unsigned char)*pos)) pos++;
        /* TStmpE: 的格式在原版中是 (int, energy) */
        if (k == 4) {
            int tmpi = 0;
            float e = 0.0f;
            if (sscanf(pos, "%d %f", &tmpi, &e) == 2) { *energy_out = e; return 1; }
        } else {
            float e = 0.0f;
            if (sscanf(pos, "%f", &e) == 1) { *energy_out = e; return 1; }
        }
    }
    return 0;
}

/* ========= 读取 XYZ（Fortran: rpip_readxyz） =========
   st_zb: 3 x (stn*nn_atom) 的连续池（下标：comp*1 + 构象基址 + 原子偏移）
*/
typedef struct {
    int    nn_atom;          /* 每个构象的原子数 */
    int    stn;              /* 构象个数 */
    float  mine;             /* 最低能量（用于筛选） */
    float *st_ee;            /* 每个构象的能量（长度 MAX_STATES） */
    float *st_zb;            /* 3 x MAX_STATES 的坐标池 */
    int    *st_zi;           /* 原子 Z（长度 MAX_ATOMS）——注意：这是“单个分子模板”的 Z；Fortran 里每个构象 Z 相同 */
} XYZSet;

static int read_xyz_all(const char *fname, XYZSet *out) {
    init_element_table();
    FILE *fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "ERROR opening %s\n", fname);
        return 0;
    }
    out->nn_atom = 0;
    out->stn = 0;
    out->mine = 9.9e30f;

    char buf[1024];
    int na = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        /* 可能是空行 / 注释，跳过 */
        char line[1024]; strcpy(line, buf);
        char *s = line; while (*s && isspace((unsigned char)*s)) s++;
        if (*s == '\0') continue;

        /* 第一行：原子数 */
        if (isdigit((unsigned char)*s)) {
            na = 0;
            if (sscanf(s, "%d", &na) != 1 || na <= 0 || na > MAX_ATOMS-2) {
                /* 不合格块——跳过 */
								for (int i=0;i<na;i++) {
								    if (fgets(buf, sizeof(buf), fp) == NULL) break;
								};                
                continue;
            }
            /* 标题行（能量） */
            if (!fgets(buf, sizeof(buf), fp)) break;
            float energy = 0.0f;
            get_energy_from_title(buf, &energy);

            /* 逐行读原子 */
            int ok = 1;
            float xyz_local[3*MAX_ATOMS];
            int   zi_local[MAX_ATOMS];
            for (int i = 0; i < na; ++i) {
                if (!fgets(buf, sizeof(buf), fp)) { ok = 0; break; }
                char sym[8]={0};
                float x,y,z;
                /* 行可能形如：C   0.1  0.2  0.3  或  6   0.1 0.2 0.3 */
                char t[1024]; strcpy(t, buf);
                char *pp = t;
                while (*pp && isspace((unsigned char)*pp)) pp++;
                /* 先看是不是数字（原子序号） */
                if (isdigit((unsigned char)*pp) || ((*pp=='-'||*pp=='+') && isdigit((unsigned char)pp[1]))) {
                    int Z=0;
                    if (sscanf(pp, "%d %f %f %f", &Z, &x, &y, &z) != 4) { ok = 0; break; }
                    if (Z<=0 || Z>120) { ok = 0; break; }
                    zi_local[i] = Z;
                } else {
                    if (sscanf(pp, "%2s %f %f %f", sym, &x, &y, &z) != 4) { ok = 0; break; }
                    int Z = elem_symbol_to_Z(sym);
                    if (Z<=0) { ok = 0; break; }
                    zi_local[i] = Z;
                }
                xyz_local[3*i+0] = x;
                xyz_local[3*i+1] = y;
                xyz_local[3*i+2] = z;
            }
            if (!ok) break;

            /* 记录第一次的原子表（Fortran: tmp_zi1 -> st_zi） */
            if (out->stn == 0) {
                out->nn_atom = na;
                for (int i=0;i<na;i++) out->st_zi[i] = zi_local[i];
            }

            if (energy > out->mine + glob_thresh_e3) {
                /* 过高能量的构象，跳过 */
                continue;
            }
            if (energy < out->mine) out->mine = energy;

            /* 存入 st_ee / st_zb */
            int stn = out->stn;
            if (stn >= MAX_STATES) { fprintf(stderr,"Too many states\n"); break; }
            out->st_ee[stn] = energy;
            long base = (long)stn * out->nn_atom; /* 每个构象的原子数相同 */
            for (int i=0;i<na;i++) {
                long j = base + i;
                out->st_zb[3*j+0] = xyz_local[3*i+0];
                out->st_zb[3*j+1] = xyz_local[3*i+1];
                out->st_zb[3*j+2] = xyz_local[3*i+2];
            }
            out->stn++;
        } else {
            /* 不识别，继续 */
            continue;
        }
    }
    fclose(fp);
    printf("%s  nn_atom=%d  stn=%d  mine=%g\n", fname, out->nn_atom, out->stn, out->mine);
    return 1;
}

/* ========= 旋转对齐（Fortran: sf_rotz_zb）
   目标：把分子整体平移使 ii1 到原点；再旋转让 (ii2 - ii1) 对齐到 -Z 方向
*/
static void sf_rotz_zb_align(int nn_atom, float *zb /*3*nn_atom*/, int ii1 /*0-based*/, int ii2 /*0-based*/) {
    if (ii1<0 || ii1>=nn_atom || ii2<0 || ii2>=nn_atom) return;

    /* 平移：ii1 到原点 */
    float t0x = zb[3*ii1+0], t0y = zb[3*ii1+1], t0z = zb[3*ii1+2];
    for (int i=0;i<nn_atom;i++) {
        zb[3*i+0] -= t0x;
        zb[3*i+1] -= t0y;
        zb[3*i+2] -= t0z;
    }

    /* 第一步：绕 Z 旋转，把向量在 XY 平面投影对齐 X 正向 */
    float v1x = zb[3*ii2+0], v1y = zb[3*ii2+1], v1z = zb[3*ii2+2];
    float vx = v1x, vy = v1y, vz = 0.0f;
    float r = sqrtf(vx*vx + vy*vy + 1e-9f);
    vx /= r; vy /= r; /* 单位化 */
    float cq = vx * 1.0f + vy * 0.0f; /* 与 +X 方向夹角的 cos */
    float q1 = -acosf(cq);
    /* 原代码带一个“更小绝对值”的选择，这里复现：比较两种方向后的投影绝对值 */
    float test1 = fabsf( cosf(-q1)*v1y + sinf(-q1)*v1x );
    float test2 = fabsf( cosf( q1)*v1y + sinf( q1)*v1x );
    if (test1 < test2) q1 = -q1;

    for (int i=0;i<nn_atom;i++) {
        float x = zb[3*i+0], y = zb[3*i+1], z = zb[3*i+2];
        float nx = cosf(q1)*x - sinf(q1)*y;
        float ny = cosf(q1)*y + sinf(q1)*x;
        zb[3*i+0] = nx; zb[3*i+1] = ny; zb[3*i+2] = z;
    }

    /* 第二步：绕 Y 旋转，使得 (ii2) 向量指向 -Z */
    float ax = zb[3*ii2+0], ay = 0.0f, az = zb[3*ii2+2];
    r = sqrtf(ax*ax + ay*ay + az*az) + 1e-12f;
    ax /= r; az /= r;
    /* 目标方向 (0,0,-1) 与 (ax,0,az) 的夹角余弦 */
    cq = ax*0.0f + az*(-1.0f);
    float q2 = -acosf(cq);
    float t1 = fabsf( cosf(-q2)*zb[3*ii2+2] + sinf(-q2)*zb[3*ii2+0] );
    float t2 = fabsf( cosf( q2)*zb[3*ii2+2] + sinf( q2)*zb[3*ii2+0] );
    if (t1 > t2) q2 = -q2;

    for (int i=0;i<nn_atom;i++) {
        float x = zb[3*i+0], y = zb[3*i+1], z = zb[3*i+2];
        float nx = cosf(q2)*x - sinf(q2)*z;
        float nz = cosf(q2)*z + sinf(q2)*x;
        zb[3*i+0] = nx; zb[3*i+1] = y; zb[3*i+2] = nz;
    }
}

/* ========= 旋转能（Fortran: sf_rot_energy_zb）
   输入：分子1/2 原子数、Z 表、坐标（3*na）、需排除的连接原子索引 iii1/iii2
   输出：energy（kcal/mol，上限裁剪到 5.0）
		原子	σ (?)	ε (kcal/mol)
		H	2.571	0.044
		He	2.104	0.056
		Li	2.183	0.025
		Be	2.445	0.085
		B	3.239	0.180
		C	3.056	0.105
		N	2.904	0.069
		O	2.777	0.060
		F	2.670	0.050
		Ne	2.573	0.042
		Na	2.657	0.030
		Mg	2.691	0.111
		Al	4.007	0.505
		Si	3.826	0.402
		P	3.694	0.305
		S	3.594	0.274
		Cl	3.516	0.227
		Ar	3.445	0.185

*/
/* ========= 修复：rot_energy_vdw 函数 - 统一使用原子单位 ========= */
/* ========= 修正后的rot_energy_vdw函数 - 包含正确的电荷分配 ========= */
static void rot_energy_vdw(int na1, int na2, const int *zi1, const int *zi2,
                           const float *zb1, const float *zb2, float *energy,
                           int iii1, int iii2)
{
    /* 简化 LJ 参数，使用原子单位 */
    float atom_epsi[121]; float atom_rmin[121];
    for (int i=0;i<121;i++){ 
        atom_epsi[i] = -0.02f * KCAL_TO_AU;  /* 转换为a.u. */
        atom_rmin[i] = 2.00f; 
    }
    
    /* 原子参数 - 全部转换为原子单位 */
    atom_epsi[17] = -0.227f * KCAL_TO_AU; atom_rmin[17] = 1.76f;  /* Cl */
    atom_epsi[16] = -0.274f * KCAL_TO_AU; atom_rmin[16] = 1.78f;  /* S */
    atom_epsi[15] = -0.305f * KCAL_TO_AU; atom_rmin[15] = 1.84f;  /* P */
    atom_epsi[14] = -0.402f * KCAL_TO_AU; atom_rmin[14] = 1.92f;  /* Si */
    atom_epsi[ 9] = -0.050f * KCAL_TO_AU; atom_rmin[ 9] = 1.35f;  /* F */
    atom_epsi[ 8] = -0.060f * KCAL_TO_AU; atom_rmin[ 8] = 1.39f;  /* O */
    atom_epsi[ 7] = -0.069f * KCAL_TO_AU; atom_rmin[ 7] = 1.45f;  /* N */
    atom_epsi[ 6] = -0.105f * KCAL_TO_AU; atom_rmin[ 6] = 1.52f;  /* C */
    atom_epsi[ 5] = -0.180f * KCAL_TO_AU; atom_rmin[ 5] = 1.62f;  /* B */
    atom_epsi[ 1] = -0.022f * KCAL_TO_AU; atom_rmin[ 1] = 1.28f;  /* H */

    float e0 = 0.0f;

    if (na1>0 && na2>0) {
        /* 第一轮：范德华（cutoff 4.5A） */
        for (int i=0;i<na1;i++) {
            if (i == iii1) continue;
            for (int j=0;j<na2;j++) {
                if (j == iii2) continue;
                float dx = zb1[3*i+0] - zb2[3*j+0]; if (dx > 4.5f || dx < -4.5f) continue;
                float dy = zb1[3*i+1] - zb2[3*j+1]; if (dy > 4.5f || dy < -4.5f) continue;
                float dz = zb1[3*i+2] - zb2[3*j+2]; if (dz > 4.5f || dz < -4.5f) continue;
                float r1 = sqrtf(dx*dx + dy*dy + dz*dz); if (r1 > 4.5f) continue;

                int m1 = zi1[i], m2 = zi2[j];
                if (m1==0 || m2==0) {
                    if (r1 < 1.0f) e0 += 2.0f * KCAL_TO_AU;
                    else if (r1 < 1.5f) e0 += 0.1f * KCAL_TO_AU;
                    else if (r1 < 1.7f) e0 += 0.01f * KCAL_TO_AU;
                    continue;
                }
                
                if (r1 < 0.7f) e0 += 80.0f * KCAL_TO_AU;
                else if (r1 < 1.0f) e0 += 30.0f * KCAL_TO_AU;
                else if (r1 < 1.7f) e0 += 2.0f * KCAL_TO_AU;
                
                float r6 = r1*r1*r1; r6 *= r6;
                float rmin = atom_rmin[m1] + atom_rmin[m2];
                float rmin6 = rmin*rmin*rmin; rmin6 *= rmin6;
                float eps = 0.535f * sqrtf(fabsf(atom_epsi[m1]*atom_epsi[m2]));
                float c1 = eps * (rmin6*rmin6);
                float c2 = -2.0f * eps * (rmin6);
                float de1 = 0.5f * c1 / (r6*r6);
                float de2 = 0.5f * c2 / (r6);
                e0 += de1 + de2;
                if (e0 > 30.0f * KCAL_TO_AU) break;
            }
            if (e0 > 30.0f * KCAL_TO_AU) break;
        }

        /* 第二轮：库仑项（1.6A 截断到 3.5A） */
        for (int i=0;i<na1;i++) {
            if (i == iii1) continue;
            for (int j=0;j<na2;j++) {
                if (j == iii2) continue;
                float dx = zb1[3*i+0] - zb2[3*j+0]; if (dx > 3.5f || dx < -3.5f) continue;
                float dy = zb1[3*i+1] - zb2[3*j+1]; if (dy > 3.5f || dy < -3.5f) continue;
                float dz = zb1[3*i+2] - zb2[3*j+2]; if (dz > 3.5f || dz < -3.5f) continue;
                float r1 = sqrtf(dx*dx + dy*dy + dz*dz); if (r1 > 3.5f) continue;

                int m1 = zi1[i], m2 = zi2[j];
                if (m1==0 || m2==0) continue;
                
                /* 根据原子类型分配电荷 */
                float q1 = 0.0f, q2 = 0.0f;
                
                /* 原子1的电荷分配 */
                switch (m1) {
                    case 1:  q1 = +0.03f; break;  /* H */
                    case 5:  q1 = +0.01f; break;  /* B */
                    case 6:  q1 = +0.01f; break;  /* C */
                    case 7:  q1 = -0.03f; break;  /* N */
                    case 8:  q1 = -0.05f; break;  /* O */
                    case 9:  q1 = -0.01f; break;  /* F */
                    case 14: q1 = +0.01f; break;  /* Si */
                    case 15: q1 = -0.02f; break;  /* P */
                    case 16: q1 = -0.03f; break;  /* S */
                    case 17: q1 = -0.02f; break;  /* Cl */
                    default: q1 = 0.00f; break;   /* 其他元素 */
                }
                
                /* 原子2的电荷分配 */
                switch (m2) {
                    case 1:  q2 = +0.03f; break;  /* H */
                    case 5:  q2 = +0.01f; break;  /* B */
                    case 6:  q2 = +0.01f; break;  /* C */
                    case 7:  q2 = -0.03f; break;  /* N */
                    case 8:  q2 = -0.05f; break;  /* O */
                    case 9:  q2 = -0.01f; break;  /* F */
                    case 14: q2 = +0.01f; break;  /* Si */
                    case 15: q2 = -0.02f; break;  /* P */
                    case 16: q2 = -0.03f; break;  /* S */
                    case 17: q2 = -0.02f; break;  /* Cl */
                    default: q2 = 0.00f; break;   /* 其他元素 */
                }
                
                /* 计算库仑相互作用 */
                float r_eff = (r1 > 1.6f) ? r1 : 1.6f;
                float de = 0.25f * 0.535f * q1 * q2 / r_eff;
                e0 += de;
            }
        }
    }

    /* 裁剪到上限 20.0 kcal/mol，转换为原子单位 */
    if (e0 >= 19.9f * KCAL_TO_AU) e0 = 20000.0f * KCAL_TO_AU;
    *energy = e0;  /* 直接输出原子单位 */
}


/* ========= 主拼接过程（Fortran: rpip_1x2）
   - 删除各自“第二个”指定原子（置 Z=0）
   - 对齐：mol1(ii1,ii2) 把 ii1 放原点、向量(ii1→ii2) 指向 -Z；mol2(ii2,ii1) 相反
   - 扫描旋转角，平滑偶点，找局部极小，输出
*/
/* ========= 主拼接过程（Fortran: rpip_1x2） */
/* ========= 修复：rpip_1x2 函数 - 统一能量单位 ========= */
static void rpip_1x2(const XYZSet *A, const XYZSet *B,
                     int mol1_abond_1, int mol1_abond_2,
                     int mol2_abond_1, int mol2_abond_2)
{
    FILE *fo = fopen("3.xyz", "w");
    if (!fo) { fprintf(stderr,"ERROR: cannot open 3.xyz for write\n"); return; }

    /* 准备 Z（删除"第二个"原子） */
    int zi1[MAX_ATOMS], zi2[MAX_ATOMS];
    for (int i=0;i<A->nn_atom;i++) zi1[i] = A->st_zi[i];
    for (int i=0;i<B->nn_atom;i++) zi2[i] = B->st_zi[i];
    if (mol1_abond_2 >= 1 && mol1_abond_2 <= A->nn_atom) zi1[mol1_abond_2-1] = 0;
    if (mol2_abond_2 >= 1 && mol2_abond_2 <= B->nn_atom) zi2[mol2_abond_2-1] = 0;

    float cur_dz = (glob_atom_radi[ zi1[mol1_abond_1-1] ] - glob_atom_radi[ zi2[mol2_abond_2-1] ]);

    int stn_out = 0;
    for (int s1 = 0; s1 < A->stn; ++s1) {
        float energy1 = A->st_ee[s1];  /* 假设已经是原子单位 */
        long base1 = (long)s1 * A->nn_atom;
        
        float *tmp1 = (float*)malloc(sizeof(float) * 3 * A->nn_atom);
        for (int i=0;i<A->nn_atom;i++) {
            tmp1[3*i+0] = A->st_zb[3*(base1+i)+0];
            tmp1[3*i+1] = A->st_zb[3*(base1+i)+1];
            tmp1[3*i+2] = A->st_zb[3*(base1+i)+2];
        }
        
        if (mol1_abond_2 > 0) sf_rotz_zb_align(A->nn_atom, tmp1, mol1_abond_1-1, mol1_abond_2-1);

        for (int s2 = 0; s2 < B->stn; ++s2) {
            float energy2 = B->st_ee[s2];  /* 假设已经是原子单位 */
            long base2 = (long)s2 * B->nn_atom;
            float *tmp2 = (float*)malloc(sizeof(float) * 3 * B->nn_atom);
            for (int i=0;i<B->nn_atom;i++) {
                tmp2[3*i+0] = B->st_zb[3*(base2+i)+0];
                tmp2[3*i+1] = B->st_zb[3*(base2+i)+1];
                tmp2[3*i+2] = B->st_zb[3*(base2+i)+2];
            }
            if (mol2_abond_2 > 0) sf_rotz_zb_align(B->nn_atom, tmp2, mol2_abond_2-1, mol2_abond_1-1);
            	
            for (int j=0;j<B->nn_atom;j++) {
                tmp2[3*j+2] = tmp2[3*j+2] - cur_dz*1.00f;
            }

            /* 预扫描 36 个角度 */
            float ee[37]={0}; 
            int is_local_min[37]={0};
            
            // 扫描奇数点
            for (int roti = 1; roti <= 36; roti += 2) {
                float q = DEG2RAD((roti-1) * 10.0f);
                float cos_q = cosf(q);
                float sin_q = sinf(q);
                
                float *rot2 = (float*)malloc(sizeof(float)*3*B->nn_atom);
                for (int j=0;j<B->nn_atom;j++) {
                    float x = tmp2[3*j+0], y = tmp2[3*j+1], z = tmp2[3*j+2];
                    float nx = cos_q*x - sin_q*y;
                    float ny = cos_q*y + sin_q*x;
                    float nz = z ;
                    rot2[3*j+0] = nx; rot2[3*j+1] = ny; rot2[3*j+2] = nz;
                }
                
                float E = 0.0f;
                rot_energy_vdw(A->nn_atom, B->nn_atom, zi1, zi2, tmp1, rot2, &E, 
                              mol1_abond_1-1, mol2_abond_1-1);
                ee[roti] = E;  /* 已经是原子单位 */
                free(rot2);
            }
            
            // 插值偶数点
            for (int roti=2; roti<=34; roti+=2) {
                ee[roti] = 0.5f*ee[roti-1] + 0.5f*ee[roti+1];
            }
            ee[36] = 0.5f*ee[35] + 0.5f*ee[1];
            
            // 寻找局部极小值点
            for (int roti=1; roti<=36; roti++) {
                int prev = (roti == 1) ? 36 : roti-1;
                int next = (roti == 36) ? 1 : roti+1;
                
                if (ee[roti] < ee[prev] && ee[roti] < ee[next] &&ee[roti]<20.0f*KCAL_TO_AU) {
                    is_local_min[roti] = 1;
                } else {
                    is_local_min[roti] = 0;
                }
            }

            /* 输出局部能量最低点 */
            for (int roti=1; roti<=36; ++roti) {
                if (!is_local_min[roti]) continue;
                
                float q = DEG2RAD((roti-1)*10.0f);
                float cos_q = cosf(q);
                float sin_q = sinf(q);
                float *rot2f = (float*)malloc(sizeof(float)*3*B->nn_atom);
                for (int j=0;j<B->nn_atom;j++) {
                    float x = tmp2[3*j+0], y = tmp2[3*j+1], z = tmp2[3*j+2];
                    float nx = cos_q*x - sin_q*y;
                    float ny = cos_q*y + sin_q*x;
                    float nz = z ;
                    rot2f[3*j+0] = nx; rot2f[3*j+1] = ny; rot2f[3*j+2] = nz;
                }
                
                float Evdw = 0.0f; 
                rot_energy_vdw(A->nn_atom, B->nn_atom, zi1, zi2, tmp1, rot2f, &Evdw, 
                              mol1_abond_1-1, mol2_abond_1-1);

                float e_scan = ee[roti];  /* 原子单位 */
                float total_energy = energy1 + energy2 + e_scan;  /* 全部原子单位 */

                /* 组合输出 */
                int nn_atom = A->nn_atom + B->nn_atom - 2;
                fprintf(fo, "%d\n", nn_atom);
                
                fprintf(fo, "Energy:%12.6f a.u. s1:%3d s2:%3d rot_angle:%3d escan:%6.4f\n", 
                        total_energy, s1+1, s2+1, (roti-1)*10,e_scan);

                /* 分子1原子 */
                for (int j=0;j<A->nn_atom;j++) {
                    if (zi1[j] <= 0) continue;
                    int Z = zi1[j];
                    fprintf(fo, "%-2s%12.6f%12.6f%12.6f\n",
                            glob_ele_table[Z],
                            tmp1[3*j+0], tmp1[3*j+1], tmp1[3*j+2]);
                }
                /* 分子2原子 */
                for (int j=0;j<B->nn_atom;j++) {
                    if (zi2[j] <= 0) continue;
                    int Z = zi2[j];
                    fprintf(fo, "%-2s%12.6f%12.6f%12.6f\n",
                            glob_ele_table[Z],
                            rot2f[3*j+0], rot2f[3*j+1], rot2f[3*j+2] );
                }
                stn_out++;
                free(rot2f);
            }
            free(tmp2);
        }
        free(tmp1);
    }
    fclose(fo);
    printf("mol_1x2 done. Output conformers: %d\n", stn_out);
}

/* ========= 定义输出结构 ========= */
typedef struct {
    float total_energy;      // 总能量（原子单位）
    float energy1;           // 分子1的能量
    float energy2;           // 分子2的能量  
    float interaction_energy; // 相互作用能（原子单位）
    int rot_angle;           // 旋转角度（度）
    int s1_index;            // 分子1的构象索引
    int s2_index;            // 分子2的构象索引
    int nn_atom;             // 总原子数
    int *zi;                 // 原子类型（长度nn_atom）
    float *zb;               // 坐标（3 * nn_atom）
    char energy_line[256];   // 能量标签行内容
} CombinedConformer;

typedef struct {
    int n_conformers;              // 构象数量
    CombinedConformer *conformers; // 构象数组
} CombinedResult;

/* ========= 比较函数，用于能量排序 ========= */
static int compare_energy(const void *a, const void *b) {
    const CombinedConformer *ca = (const CombinedConformer *)a;
    const CombinedConformer *cb = (const CombinedConformer *)b;
    if (ca->total_energy < cb->total_energy) return -1;
    if (ca->total_energy > cb->total_energy) return 1;
    return 0;
}

/* ========= 释放CombinedResult内存 ========= */
void free_combined_result(CombinedResult *result) {
    if (result && result->conformers) {
        for (int i = 0; i < result->n_conformers; i++) {
            if (result->conformers[i].zi) free(result->conformers[i].zi);
            if (result->conformers[i].zb) free(result->conformers[i].zb);
        }
        free(result->conformers);
        result->conformers = NULL;
        result->n_conformers = 0;
    }
}

/* ========= 新版主拼接过程，返回内存中的结果 ========= */
/* ========= 修正后的主拼接过程（返回内存中的结果） ========= */
static CombinedResult rpip_1x2m(const XYZSet *A, const XYZSet *B,
                               int mol1_abond_1, int mol1_abond_2,
                               int mol2_abond_1, int mol2_abond_2,
                               int max_add_outn)
{
    CombinedResult result = {0, NULL};
    
    /* 计算最大输出数量 */
    int max_outn = A->stn + max_add_outn;
    if (max_outn > MAX_STATES) max_outn = MAX_STATES;
    
    /* 分配输出数组 */
    result.conformers = (CombinedConformer*)calloc(max_outn, sizeof(CombinedConformer));
    if (!result.conformers) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return result;
    }

    /* 准备 Z（删除"第二个"原子） */
    int zi1[MAX_ATOMS], zi2[MAX_ATOMS];
    for (int i=0;i<A->nn_atom;i++) zi1[i] = A->st_zi[i];
    for (int i=0;i<B->nn_atom;i++) zi2[i] = B->st_zi[i];
    if (mol1_abond_2 >= 1 && mol1_abond_2 <= A->nn_atom) zi1[mol1_abond_2-1] = 0;
    if (mol2_abond_2 >= 1 && mol2_abond_2 <= B->nn_atom) zi2[mol2_abond_2-1] = 0;

    /* cur_dz：两个"首原子"的原子半径差 - 与 rpip_1x2 保持一致 */
    float cur_dz = (glob_atom_radi[ zi1[mol1_abond_1-1] ] - glob_atom_radi[ zi2[mol2_abond_2-1] ]);

    /* 为每个s1存储多个候选结构 */
    CombinedConformer **s1_candidates = (CombinedConformer**)calloc(A->stn, sizeof(CombinedConformer*));
    int *s1_candidate_count = (int*)calloc(A->stn, sizeof(int));
    int *s1_candidate_capacity = (int*)calloc(A->stn, sizeof(int));
    
    /* 初始化每个s1的候选数组 */
    for (int s1 = 0; s1 < A->stn; s1++) {
        s1_candidate_capacity[s1] = 10; // 初始容量
        s1_candidates[s1] = (CombinedConformer*)calloc(s1_candidate_capacity[s1], sizeof(CombinedConformer));
        s1_candidate_count[s1] = 0;
    }

    /* 主循环：处理每个s1和s2的组合 */
    for (int s1 = 0; s1 < A->stn; ++s1) {
        float energy1 = A->st_ee[s1];
        long base1 = (long)s1 * A->nn_atom;
        
        /* 拷贝此构象坐标 */
        float *tmp1 = (float*)malloc(sizeof(float) * 3 * A->nn_atom);
        for (int i=0;i<A->nn_atom;i++) {
            tmp1[3*i+0] = A->st_zb[3*(base1+i)+0];
            tmp1[3*i+1] = A->st_zb[3*(base1+i)+1];
            tmp1[3*i+2] = A->st_zb[3*(base1+i)+2];
        }
        
        /* 对齐到 (ii1->ii2) 指向 -Z */
        if (mol1_abond_2 > 0) sf_rotz_zb_align(A->nn_atom, tmp1, mol1_abond_1-1, mol1_abond_2-1);

        for (int s2 = 0; s2 < B->stn; ++s2) {
            float energy2 = B->st_ee[s2];
            long base2 = (long)s2 * B->nn_atom;
            float *tmp2 = (float*)malloc(sizeof(float) * 3 * B->nn_atom);
            for (int i=0;i<B->nn_atom;i++) {
                tmp2[3*i+0] = B->st_zb[3*(base2+i)+0];
                tmp2[3*i+1] = B->st_zb[3*(base2+i)+1];
                tmp2[3*i+2] = B->st_zb[3*(base2+i)+2];
            }
            if (mol2_abond_2 > 0) sf_rotz_zb_align(B->nn_atom, tmp2, mol2_abond_2-1, mol2_abond_1-1);
            
            /* 与rpip_1x2一致的Z轴位移 */
            for (int j=0;j<B->nn_atom;j++) {
                tmp2[3*j+2] = tmp2[3*j+2] - cur_dz * 1.00f;
            }

            /* 预扫描 36 个角度，记录能量并对偶点做平滑 */
            float ee[37]={0}; 
            int is_local_min[37]={0};
            
            // 扫描奇数点
            for (int roti = 1; roti <= 36; roti += 2) {
                float q = DEG2RAD((roti-1) * 10.0f);
                float cos_q = cosf(q);
                float sin_q = sinf(q);
                
                float *rot2 = (float*)malloc(sizeof(float)*3*B->nn_atom);
                for (int j=0;j<B->nn_atom;j++) {
                    float x = tmp2[3*j+0], y = tmp2[3*j+1], z = tmp2[3*j+2];
                    float nx = cos_q*x - sin_q*y;
                    float ny = cos_q*y + sin_q*x;
                    float nz = z;
                    rot2[3*j+0] = nx; rot2[3*j+1] = ny; rot2[3*j+2] = nz;
                }
                
                float E = 0.0f;
                rot_energy_vdw(A->nn_atom, B->nn_atom, zi1, zi2, tmp1, rot2, &E, 
                              mol1_abond_1-1, mol2_abond_1-1);
                ee[roti] = E;
                free(rot2);
            }
            
            // 插值偶数点
            for (int roti=2; roti<=34; roti+=2) {
                ee[roti] = 0.5f*ee[roti-1] + 0.5f*ee[roti+1];
            }
            ee[36] = 0.5f*ee[35] + 0.5f*ee[1];
            
            // 寻找局部极小值点
            for (int roti=1; roti<=36; roti++) {
                int prev = (roti == 1) ? 36 : roti-1;
                int next = (roti == 36) ? 1 : roti+1;
                
                if (ee[roti] < ee[prev] && ee[roti] < ee[next] && ee[roti]<20.0f*KCAL_TO_AU) {
                    is_local_min[roti] = 1;
                } else {
                    is_local_min[roti] = 0;
                }
            }

            /* 处理局部极小值点 */
            for (int roti=1; roti<=36; ++roti) {
                if (!is_local_min[roti]) continue;

                float q = DEG2RAD((roti-1)*10.0f);
                float cos_q = cosf(q);
                float sin_q = sinf(q);
                float *rot2f = (float*)malloc(sizeof(float)*3*B->nn_atom);
                for (int j=0;j<B->nn_atom;j++) {
                    float x = tmp2[3*j+0], y = tmp2[3*j+1], z = tmp2[3*j+2];
                    float nx = cos_q*x - sin_q*y;
                    float ny = cos_q*y + sin_q*x;
                    float nz = z;
                    rot2f[3*j+0] = nx; rot2f[3*j+1] = ny; rot2f[3*j+2] = nz;
                }
                
                float Evdw = 0.0f; 
                rot_energy_vdw(A->nn_atom, B->nn_atom, zi1, zi2, tmp1, rot2f, &Evdw, 
                              mol1_abond_1-1, mol2_abond_1-1);

                float e_scan = ee[roti];
                float total_energy = energy1 + energy2 + e_scan;

                /* 创建新的构象结构 */
                CombinedConformer conf;
                conf.total_energy = total_energy;
                conf.energy1 = energy1;
                conf.energy2 = energy2;
                conf.interaction_energy = e_scan;
                conf.rot_angle = (roti-1)*10;
                conf.s1_index = s1;
                conf.s2_index = s2;
                conf.nn_atom = A->nn_atom + B->nn_atom - 2;
                
                snprintf(conf.energy_line, sizeof(conf.energy_line),
                        "Energy:%12.6f a.u. s1:%3d s2:%3d rot_angle:%3d escan:%6.4f",
                        total_energy, s1+1, s2+1, (roti-1)*10, e_scan);
                
                /* 分配并拷贝原子类型和坐标 */
                conf.zi = (int*)malloc(sizeof(int) * conf.nn_atom);
                conf.zb = (float*)malloc(sizeof(float) * 3 * conf.nn_atom);
                
                int atom_idx = 0;
                // 分子1原子
                for (int j=0;j<A->nn_atom;j++) {
                    if (zi1[j] <= 0) continue;
                    conf.zi[atom_idx] = zi1[j];
                    conf.zb[3*atom_idx+0] = tmp1[3*j+0];
                    conf.zb[3*atom_idx+1] = tmp1[3*j+1];
                    conf.zb[3*atom_idx+2] = tmp1[3*j+2];
                    atom_idx++;
                }
                // 分子2原子
                for (int j=0;j<B->nn_atom;j++) {
                    if (zi2[j] <= 0) continue;
                    conf.zi[atom_idx] = zi2[j];
                    conf.zb[3*atom_idx+0] = rot2f[3*j+0];
                    conf.zb[3*atom_idx+1] = rot2f[3*j+1];
                    conf.zb[3*atom_idx+2] = rot2f[3*j+2];
                    atom_idx++;
                }

                /* 存储到s1的候选数组中 */
                int count = s1_candidate_count[s1];
                
                /* 如果数组已满，需要扩容 */
                if (count >= s1_candidate_capacity[s1]) {
                    s1_candidate_capacity[s1] *= 2;
                    CombinedConformer *new_array = (CombinedConformer*)realloc(s1_candidates[s1], 
                        s1_candidate_capacity[s1] * sizeof(CombinedConformer));
                    if (new_array) {
                        s1_candidates[s1] = new_array;
                    } else {
                        fprintf(stderr, "ERROR: Memory reallocation failed for s1_candidates\n");
                        free(rot2f);
                        continue;
                    }
                }
                
                /* 插入并保持排序（按能量升序） */
                int insert_pos = count;
                for (int i = 0; i < count; i++) {
                    if (total_energy < s1_candidates[s1][i].total_energy) {
                        insert_pos = i;
                        break;
                    }
                }
                
                /* 移动后续元素 */
                for (int i = count; i > insert_pos; i--) {
                    s1_candidates[s1][i] = s1_candidates[s1][i-1];
                }
                
                /* 插入新结构 */
                s1_candidates[s1][insert_pos] = conf;
                s1_candidate_count[s1]++;

                free(rot2f);
            }
            free(tmp2);
        }
        free(tmp1);
    }

    /* 修正输出逻辑：避免重复输出 */
    
    /* 第一阶段：收集每个s1的最低能量结构 */
    CombinedConformer *s1_best = (CombinedConformer*)calloc(A->stn, sizeof(CombinedConformer));
    int s1_best_used = 0;
    
    for (int s1 = 0; s1 < A->stn; s1++) {
        if (s1_candidate_count[s1] > 0) {
            s1_best[s1_best_used++] = s1_candidates[s1][0];
            /* 标记第一个结构已被使用 */
            s1_candidates[s1][0].zi = NULL;
            s1_candidates[s1][0].zb = NULL;
        }
    }
    
    /* 第二阶段：收集所有剩余的候选结构 */
    CombinedConformer *remaining_candidates = NULL;
    int remaining_count = 0;
    int remaining_capacity = 0;
    
    /* 计算剩余候选结构的总数 */
    for (int s1 = 0; s1 < A->stn; s1++) {
        for (int i = 0; i < s1_candidate_count[s1]; i++) {
            if (s1_candidates[s1][i].zi != NULL) { /* 未被使用 */
                remaining_capacity++;
            }
        }
    }
    
    if (remaining_capacity > 0) {
        remaining_candidates = (CombinedConformer*)calloc(remaining_capacity, sizeof(CombinedConformer));
        
        for (int s1 = 0; s1 < A->stn; s1++) {
            for (int i = 0; i < s1_candidate_count[s1]; i++) {
                if (s1_candidates[s1][i].zi != NULL) { /* 未被使用 */
                    if (remaining_count < remaining_capacity) {
                        remaining_candidates[remaining_count++] = s1_candidates[s1][i];
                        /* 标记为已转移，避免重复释放 */
                        s1_candidates[s1][i].zi = NULL;
                        s1_candidates[s1][i].zb = NULL;
                    }
                }
            }
        }
    }
    
    /* 对剩余候选结构按能量排序 */
    if (remaining_count > 0) {
        qsort(remaining_candidates, remaining_count, sizeof(CombinedConformer), compare_energy);
    }
    
    /* 构建最终输出 */
    int out_idx = 0;
    
    /* 首先添加每个s1的最佳结构 */
    for (int i = 0; i < s1_best_used && out_idx < max_outn; i++) {
        result.conformers[out_idx++] = s1_best[i];
    }
    
    /* 然后添加剩余的最佳候选结构 */
    if (out_idx < max_outn && remaining_count > 0) {
        int add_count = max_outn - out_idx;
        if (add_count > remaining_count) {
            add_count = remaining_count;
        }
        
        for (int i = 0; i < add_count; i++) {
            result.conformers[out_idx++] = remaining_candidates[i];
            remaining_candidates[i].zi = NULL; /* 标记为已使用 */
            remaining_candidates[i].zb = NULL;
        }
    }
    
    result.n_conformers = out_idx;
    
    /* 最终排序：所有输出结构按能量排序 */
    if (result.n_conformers > 0) {
        qsort(result.conformers, result.n_conformers, sizeof(CombinedConformer), compare_energy);
    }

    /* 清理内存 - 修正释放逻辑 */
    
    /* 释放s1_best数组中未使用的结构 */
    for (int i = out_idx; i < s1_best_used; i++) {
        if (s1_best[i].zi) {
            free(s1_best[i].zi);
            free(s1_best[i].zb);
        }
    }
    free(s1_best);
    
    /* 释放remaining_candidates中未使用的结构 */
    if (remaining_candidates) {
        for (int i = 0; i < remaining_count; i++) {
            if (remaining_candidates[i].zi) {
                free(remaining_candidates[i].zi);
                free(remaining_candidates[i].zb);
            }
        }
        free(remaining_candidates);
    }
    
    /* 释放s1_candidates中所有未被转移的结构 */
    for (int s1 = 0; s1 < A->stn; s1++) {
        for (int i = 0; i < s1_candidate_count[s1]; i++) {
            if (s1_candidates[s1][i].zi) {
                free(s1_candidates[s1][i].zi);
                free(s1_candidates[s1][i].zb);
            }
        }
        free(s1_candidates[s1]);
    }
    
    free(s1_candidates);
    free(s1_candidate_count);
    free(s1_candidate_capacity);

    printf("mol_1x2m done. Output conformers: %d (max_outn=%d)\n", result.n_conformers, max_outn);
    return result;
}


/* ========= 主程序（Fortran: program peptide2_tmp1x2） ========= */
int main(void) {
    init_element_table();
    printf("pep1x2[C]: Hello World!!! --- from C port\n");

    /* 分配"巨大"池（与 Fortran 上限一致） */
    float *st_zb1 = (float*)calloc(3*MAX_STATES, sizeof(float));
    float *st_zb2 = (float*)calloc(3*MAX_STATES, sizeof(float));
    float *st_ee1 = (float*)calloc(MAX_STATES, sizeof(float));
    float *st_ee2 = (float*)calloc(MAX_STATES, sizeof(float));
    int   *st_zi1 = (int*)calloc(MAX_ATOMS, sizeof(int));
    int   *st_zi2 = (int*)calloc(MAX_ATOMS, sizeof(int));

    XYZSet A = {0,0,0.0f, st_ee1, st_zb1, st_zi1};
    XYZSet B = {0,0,0.0f, st_ee2, st_zb2, st_zi2};

    /* 与 Fortran 默认相同的连接定义（可按需修改或从命令行读取） */
    int mol1_abond[2] = {17,22};  /* 1-based */
    //int mol1_abond[2] = {18,23};  /* 1-based */
    //int mol1_abond[2] = {19,24};  /* 1-based */
    //int mol2_abond[2] = {18,24};  /* 1-based */
    int mol2_abond[2] = {19,25};  /* 1-based */

    /* 阈值初始化（与 Fortran 一致） */
    glob_thresh_e0=glob_thresh_e1=glob_thresh_e2=glob_thresh_e3=0.16f;

    /* 读取 XYZ */
    if (!read_xyz_all("1.xyz", &A)) return 1;
    if (!read_xyz_all("2.xyz", &B)) return 1;
    printf("abond:%d,%d,%d,%d\n",mol1_abond[0], mol1_abond[1],mol2_abond[0], mol2_abond[1]);
    /* 运行拼接到文件（未排序） */
    rpip_1x2(&A, &B, mol1_abond[0], mol1_abond[1], mol2_abond[0], mol2_abond[1]);
    
    /* 运行拼接到内存 */
    CombinedResult result_mem = rpip_1x2m(&A, &B, 
                                         mol1_abond[0], mol1_abond[1],
                                         mol2_abond[0], mol2_abond[1],
                                         1000);
    
    /* 输出rpip_1x2m的结果到memory_results.xyz */
    FILE *fo_mem = fopen("memory_results.xyz", "w");
    if (fo_mem) {
        for (int i = 0; i < result_mem.n_conformers; i++) {
            CombinedConformer *conf = &result_mem.conformers[i];
            
            fprintf(fo_mem, "%d\n", conf->nn_atom);
            fprintf(fo_mem, "%s\n", conf->energy_line);
            
            int atom_idx = 0;
            for (int j = 0; j < A.nn_atom; j++) {
                if (A.st_zi[j] <= 0) continue;
                fprintf(fo_mem, "%-2s%12.6f%12.6f%12.6f\n",
                        glob_ele_table[conf->zi[atom_idx]],
                        conf->zb[3*atom_idx+0], 
                        conf->zb[3*atom_idx+1], 
                        conf->zb[3*atom_idx+2]);
                atom_idx++;
            }
            for (int j = 0; j < B.nn_atom; j++) {
                if (B.st_zi[j] <= 0) continue;
                fprintf(fo_mem, "%-2s%12.6f%12.6f%12.6f\n",
                        glob_ele_table[conf->zi[atom_idx]],
                        conf->zb[3*atom_idx+0], 
                        conf->zb[3*atom_idx+1], 
                        conf->zb[3*atom_idx+2]);
                atom_idx++;
            }
        }
        fclose(fo_mem);
        printf("Memory results written to memory_results.xyz\n");
    }
    
    
    /* 释放内存 */
    free_combined_result(&result_mem);
    free(st_zb1); free(st_zb2); free(st_ee1); free(st_ee2); free(st_zi1); free(st_zi2);
    
    printf("Normal termination of 1x2\n");
    return 0;
}