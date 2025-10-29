#ifndef MOL1X2_H
#define MOL1X2_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 定义输出结构 */
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

// XYZSet结构体定义
typedef struct {
    int    nn_atom;          /* 每个构象的原子数 */
    int    stn;              /* 构象个数 */
    float  mine;             /* 最低能量（用于筛选） */
    float *st_ee;            /* 每个构象的能量（长度 MAX_STATES） */
    float *st_zb;            /* 3 x MAX_STATES 的坐标池 */
    int    *st_zi;           /* 原子 Z（长度 MAX_ATOMS） */
} XYZSet;

// 全局阈值变量声明
extern float glob_thresh_e0, glob_thresh_e1, glob_thresh_e2, glob_thresh_e3;

/* 导出的函数声明 */
void export_init_element_table(void);
int export_elem_symbol_to_Z(const char *sym);
int export_read_xyz_all(const char *fname, XYZSet *out);
int export_read_xyz_from_string(const char *content, XYZSet *out);
int export_read_xyz_from_string_with_validation(const char *content, XYZSet *out, char *error_msg, int error_msg_len);
CombinedResult export_rpip_1x2m(const XYZSet *A, const XYZSet *B,
                               int mol1_abond_1, int mol1_abond_2,
                               int mol2_abond_1, int mol2_abond_2,
                               int max_add_outn);
void export_free_combined_result(CombinedResult *result);

// 包装函数声明
XYZSet* create_xyz_set(void);
void free_xyz_set(XYZSet* set);
int get_xyz_set_nn_atom(XYZSet* set);
int get_xyz_set_stn(XYZSet* set);
float get_xyz_set_mine(XYZSet* set);
void set_global_thresholds(float thresh);
float get_global_threshold_e0(void);

#ifdef __cplusplus
}
#endif

#endif // MOL1X2_H