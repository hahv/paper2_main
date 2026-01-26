# Prof's skip method parameters explanation

```cpp

define NET_WIDTH_FD 640   /// net width for fd
define NET_HEIGHT_FD 360  /// net height for fd
colStep = NET_WIDTH_FD / 4; //block width = 160
rowStep = NET_HEIGHT_FD / 4; //block height = 90
if (frame.cols == 1920 && frame.rows == 1080) {
    minW = 1280;
    minH = 720;
}
else {
    minW = 0.75 * frame.cols; //HD: 960, 540
    minH = 0.75 * frame.rows;
}
```

Prof confirms that there is no special reason for choosing there values,
other than they work well in his tests.