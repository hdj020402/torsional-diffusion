#include "../src/mol1x2.c"
#include <string.h>

// 导出原本为static的函数
void export_init_element_table(void) {
    init_element_table();
}

int export_elem_symbol_to_Z(const char *sym) {
    return elem_symbol_to_Z(sym);
}

int export_read_xyz_all(const char *fname, XYZSet *out) {
    return read_xyz_all(fname, out);
}

// 从字符串读取XYZ数据的新函数
int export_read_xyz_from_string(const char *content, XYZSet *out) {
    init_element_table();
    
    // 检查输入参数
    if (!content || !out) {
        return 0;
    }
    
    out->nn_atom = 0;
    out->stn = 0;
    out->mine = 9.9e30f;

    char buf[1024];
    int na = 0;
    int pos = 0;
    int content_len = strlen(content);
    
    // 模拟文件读取过程，但使用字符串
    while (pos < content_len) {
        // 提取一行
        int line_start = pos;
        while (pos < content_len && content[pos] != '\n' && content[pos] != '\r') {
            pos++;
        }
        
        // 复制行内容
        int line_len = pos - line_start;
        if (line_len >= (int)sizeof(buf)) {
            line_len = sizeof(buf) - 1;
        }
        if (line_len > 0) {
            strncpy(buf, content + line_start, line_len);
            buf[line_len] = '\0';
        } else {
            buf[0] = '\0';
        }
        
        // 跳过换行符
        if (pos < content_len && content[pos] == '\r') pos++;
        if (pos < content_len && content[pos] == '\n') pos++;

        // 可能是空行 / 注释，跳过
        char line[1024]; strcpy(line, buf);
        char *s = line; while (*s && isspace((unsigned char)*s)) s++;
        if (*s == '\0') continue;

        // 第一行：原子数
        if (isdigit((unsigned char)*s)) {
            na = 0;
            if (sscanf(s, "%d", &na) != 1 || na <= 0 || na > MAX_ATOMS-2) {
                // 不合格块——跳过
                // 跳过接下来的na+1行（原子行+能量行）
                for (int i = 0; i < na + 1 && pos < content_len; i++) {
                    while (pos < content_len && content[pos] != '\n' && content[pos] != '\r') {
                        pos++;
                    }
                    if (pos < content_len && content[pos] == '\r') pos++;
                    if (pos < content_len && content[pos] == '\n') pos++;
                }
                continue;
            }
            
            // 标题行（能量）
            // 提取能量行
            line_start = pos;
            while (pos < content_len && content[pos] != '\n' && content[pos] != '\r') {
                pos++;
            }
            
            line_len = pos - line_start;
            if (line_len >= (int)sizeof(buf)) {
                line_len = sizeof(buf) - 1;
            }
            if (line_len > 0) {
                strncpy(buf, content + line_start, line_len);
                buf[line_len] = '\0';
            } else {
                buf[0] = '\0';
            }
            
            // 跳过换行符
            if (pos < content_len && content[pos] == '\r') pos++;
            if (pos < content_len && content[pos] == '\n') pos++;

            float energy = 0.0f;
            get_energy_from_title(buf, &energy);

            // 逐行读原子
            int ok = 1;
            float xyz_local[3*MAX_ATOMS];
            int   zi_local[MAX_ATOMS];
            for (int i = 0; i < na; ++i) {
                // 提取原子行
                line_start = pos;
                while (pos < content_len && content[pos] != '\n' && content[pos] != '\r') {
                    pos++;
                }
                
                line_len = pos - line_start;
                if (line_len >= (int)sizeof(buf)) {
                    line_len = sizeof(buf) - 1;
                }
                if (line_len > 0) {
                    strncpy(buf, content + line_start, line_len);
                    buf[line_len] = '\0';
                } else {
                    buf[0] = '\0';
                }
                
                // 跳过换行符
                if (pos < content_len && content[pos] == '\r') pos++;
                if (pos < content_len && content[pos] == '\n') pos++;

                char sym[8]={0};
                float x,y,z;
                // 行可能形如：C   0.1  0.2  0.3  或  6   0.1 0.2 0.3
                char t[1024]; strcpy(t, buf);
                char *pp = t;
                while (*pp && isspace((unsigned char)*pp)) pp++;
                // 先看是不是数字（原子序号）
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

            // 记录第一次的原子表
            if (out->stn == 0) {
                out->nn_atom = na;
                for (int i=0;i<na;i++) out->st_zi[i] = zi_local[i];
            }

            if (energy > out->mine + glob_thresh_e3) {
                // 过高能量的构象，跳过
                continue;
            }
            if (energy < out->mine) out->mine = energy;

            // 存入 st_ee / st_zb
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
            // 不识别，继续
            continue;
        }
    }
    
    return out->stn > 0 ? 1 : 0;  // 如果至少读取了一个构象，返回成功
}

// 带验证的字符串读取函数
int export_read_xyz_from_string_with_validation(const char *content, XYZSet *out, char *error_msg, int error_msg_len) {
    // 简单验证字符串格式
    if (!content || !out) {
        if (error_msg && error_msg_len > 0) {
            strncpy(error_msg, "Invalid input parameters", error_msg_len - 1);
            error_msg[error_msg_len - 1] = '\0';
        }
        return 0;
    }
    
    // 检查是否为空字符串
    if (strlen(content) == 0) {
        if (error_msg && error_msg_len > 0) {
            strncpy(error_msg, "Empty input string", error_msg_len - 1);
            error_msg[error_msg_len - 1] = '\0';
        }
        return 0;
    }
    
    // 保存原始状态
    int original_stn = out->stn;
    
    // 调用实际的读取函数
    int result = export_read_xyz_from_string(content, out);
    
    if (!result) {
        if (error_msg && error_msg_len > 0) {
            if (out->stn == original_stn) {
                strncpy(error_msg, "No valid XYZ data found in string", error_msg_len - 1);
            } else {
                strncpy(error_msg, "Failed to parse XYZ string", error_msg_len - 1);
            }
            error_msg[error_msg_len - 1] = '\0';
        }
    } else if (error_msg && error_msg_len > 0) {
        error_msg[0] = '\0';  // 清空错误信息
    }
    
    return result;
}

CombinedResult export_rpip_1x2m(const XYZSet *A, const XYZSet *B,
                               int mol1_abond_1, int mol1_abond_2,
                               int mol2_abond_1, int mol2_abond_2,
                               int max_add_outn) {
    return rpip_1x2m(A, B, mol1_abond_1, mol1_abond_2, mol2_abond_1, mol2_abond_2, max_add_outn);
}

void export_free_combined_result(CombinedResult *result) {
    free_combined_result(result);
}

// 包装函数，用于创建XYZSet结构体
XYZSet* create_xyz_set(void) {
    XYZSet* set = (XYZSet*)malloc(sizeof(XYZSet));
    if (set) {
        set->nn_atom = 0;
        set->stn = 0;
        set->mine = 9.9e30f;
        set->st_ee = (float*)calloc(MAX_STATES, sizeof(float));
        set->st_zb = (float*)calloc(3*MAX_STATES, sizeof(float));
        set->st_zi = (int*)calloc(MAX_ATOMS, sizeof(int));
    }
    return set;
}

// 释放XYZSet结构体
void free_xyz_set(XYZSet* set) {
    if (set) {
        if (set->st_ee) free(set->st_ee);
        if (set->st_zb) free(set->st_zb);
        if (set->st_zi) free(set->st_zi);
        free(set);
    }
}

// 包装函数，用于访问XYZSet的字段
int get_xyz_set_nn_atom(XYZSet* set) {
    if (set) {
        return set->nn_atom;
    }
    return 0;
}

int get_xyz_set_stn(XYZSet* set) {
    if (set) {
        return set->stn;
    }
    return 0;
}

float get_xyz_set_mine(XYZSet* set) {
    if (set) {
        return set->mine;
    }
    return 0.0f;
}

// 设置全局阈值
void set_global_thresholds(float thresh) {
    glob_thresh_e0 = glob_thresh_e1 = glob_thresh_e2 = glob_thresh_e3 = thresh;
}

// 获取全局阈值
float get_global_threshold_e0(void) {
    return glob_thresh_e0;
}