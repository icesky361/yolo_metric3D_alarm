# 在内存清理后添加详细内存日志
logger.debug(f'[内存快照] 已释放批次 {batch_num} 内存: 释放前 {mem_before/1024**2:.2f}MB → 释放后 {mem_after/1024**2:.2f}MB')