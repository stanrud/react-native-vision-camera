(window.webpackJsonp=window.webpackJsonp||[]).push([[5],{109:function(e,a,t){"use strict";t.d(a,"a",(function(){return p})),t.d(a,"b",(function(){return u}));var n=t(0),i=t.n(n);function c(e,a,t){return a in e?Object.defineProperty(e,a,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[a]=t,e}function r(e,a){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);a&&(n=n.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),t.push.apply(t,n)}return t}function b(e){for(var a=1;a<arguments.length;a++){var t=null!=arguments[a]?arguments[a]:{};a%2?r(Object(t),!0).forEach((function(a){c(e,a,t[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):r(Object(t)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(t,a))}))}return e}function o(e,a){if(null==e)return{};var t,n,i=function(e,a){if(null==e)return{};var t,n,i={},c=Object.keys(e);for(n=0;n<c.length;n++)t=c[n],a.indexOf(t)>=0||(i[t]=e[t]);return i}(e,a);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);for(n=0;n<c.length;n++)t=c[n],a.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var m=i.a.createContext({}),l=function(e){var a=i.a.useContext(m),t=a;return e&&(t="function"==typeof e?e(a):b(b({},a),e)),t},p=function(e){var a=l(e.components);return i.a.createElement(m.Provider,{value:a},e.children)},s={inlineCode:"code",wrapper:function(e){var a=e.children;return i.a.createElement(i.a.Fragment,{},a)}},d=i.a.forwardRef((function(e,a){var t=e.components,n=e.mdxType,c=e.originalType,r=e.parentName,m=o(e,["components","mdxType","originalType","parentName"]),p=l(t),d=n,u=p["".concat(r,".").concat(d)]||p[d]||s[d]||c;return t?i.a.createElement(u,b(b({ref:a},m),{},{components:t})):i.a.createElement(u,b({ref:a},m))}));function u(e,a){var t=arguments,n=a&&a.mdxType;if("string"==typeof e||n){var c=t.length,r=new Array(c);r[0]=d;var b={};for(var o in a)hasOwnProperty.call(a,o)&&(b[o]=a[o]);b.originalType=e,b.mdxType="string"==typeof e?e:n,r[1]=b;for(var m=2;m<c;m++)r[m]=t[m];return i.a.createElement.apply(null,r)}return i.a.createElement.apply(null,t)}d.displayName="MDXCreateElement"},65:function(e,a,t){"use strict";t.r(a),t.d(a,"frontMatter",(function(){return r})),t.d(a,"metadata",(function(){return b})),t.d(a,"toc",(function(){return o})),t.d(a,"default",(function(){return l}));var n=t(3),i=t(7),c=(t(0),t(109)),r={id:"cameradevice",title:"Module: CameraDevice",sidebar_label:"CameraDevice",custom_edit_url:null,hide_title:!0},b={unversionedId:"api/modules/cameradevice",id:"api/modules/cameradevice",isDocsHomePage:!1,title:"Module: CameraDevice",description:"Module: CameraDevice",source:"@site/docs/api/modules/cameradevice.md",slug:"/api/modules/cameradevice",permalink:"/react-native-vision-camera/docs/api/modules/cameradevice",editUrl:null,version:"current",sidebar_label:"CameraDevice",sidebar:"someSidebar",previous:{title:"Module: CameraCodec",permalink:"/react-native-vision-camera/docs/api/modules/cameracodec"},next:{title:"Module: CameraError",permalink:"/react-native-vision-camera/docs/api/modules/cameraerror"}},o=[{value:"Type aliases",id:"type-aliases",children:[{value:"AutoFocusSystem",id:"autofocussystem",children:[]},{value:"CameraDevice",id:"cameradevice",children:[]},{value:"CameraDeviceFormat",id:"cameradeviceformat",children:[]},{value:"ColorSpace",id:"colorspace",children:[]},{value:"FrameRateRange",id:"frameraterange",children:[]},{value:"LogicalCameraDeviceType",id:"logicalcameradevicetype",children:[]},{value:"PhysicalCameraDeviceType",id:"physicalcameradevicetype",children:[]},{value:"VideoStabilizationMode",id:"videostabilizationmode",children:[]}]},{value:"Functions",id:"functions",children:[{value:"parsePhysicalDeviceTypes",id:"parsephysicaldevicetypes",children:[]}]}],m={toc:o};function l(e){var a=e.components,t=Object(i.a)(e,["components"]);return Object(c.b)("wrapper",Object(n.a)({},m,t,{components:a,mdxType:"MDXLayout"}),Object(c.b)("h1",{id:"module-cameradevice"},"Module: CameraDevice"),Object(c.b)("h2",{id:"type-aliases"},"Type aliases"),Object(c.b)("h3",{id:"autofocussystem"},"AutoFocusSystem"),Object(c.b)("p",null,"\u01ac ",Object(c.b)("strong",{parentName:"p"},"AutoFocusSystem"),": ",Object(c.b)("em",{parentName:"p"},"contrast-detection")," ","|"," ",Object(c.b)("em",{parentName:"p"},"phase-detection")," ","|"," ",Object(c.b)("em",{parentName:"p"},"none")),Object(c.b)("p",null,"Indicates a format's autofocus system."),Object(c.b)("ul",null,Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"none"'),": Indicates that autofocus is not available"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"contrast-detection"'),": Indicates that autofocus is achieved by contrast detection. Contrast detection performs a focus scan to find the optimal position"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"phase-detection"'),": Indicates that autofocus is achieved by phase detection. Phase detection has the ability to achieve focus in many cases without a focus scan. Phase detection autofocus is typically less visually intrusive than contrast detection autofocus")),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L64"},"CameraDevice.ts:64")),Object(c.b)("hr",null),Object(c.b)("h3",{id:"cameradevice"},"CameraDevice"),Object(c.b)("p",null,"\u01ac ",Object(c.b)("strong",{parentName:"p"},"CameraDevice"),": ",Object(c.b)("em",{parentName:"p"},"Readonly"),"<{ ",Object(c.b)("inlineCode",{parentName:"p"},"devices"),": ",Object(c.b)("a",{parentName:"p",href:"/react-native-vision-camera/docs/api/modules/cameradevice#physicalcameradevicetype"},Object(c.b)("em",{parentName:"a"},"PhysicalCameraDeviceType")),"[] ; ",Object(c.b)("inlineCode",{parentName:"p"},"formats"),": ",Object(c.b)("a",{parentName:"p",href:"/react-native-vision-camera/docs/api/modules/cameradevice#cameradeviceformat"},Object(c.b)("em",{parentName:"a"},"CameraDeviceFormat")),"[] ; ",Object(c.b)("inlineCode",{parentName:"p"},"hasFlash"),": ",Object(c.b)("em",{parentName:"p"},"boolean")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"hasTorch"),": ",Object(c.b)("em",{parentName:"p"},"boolean")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"id"),": ",Object(c.b)("em",{parentName:"p"},"string")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"isMultiCam"),": ",Object(c.b)("em",{parentName:"p"},"boolean")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"maxZoom"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"minZoom"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"name"),": ",Object(c.b)("em",{parentName:"p"},"string")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"neutralZoom"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"position"),": ",Object(c.b)("a",{parentName:"p",href:"/react-native-vision-camera/docs/api/modules/cameraposition#cameraposition"},Object(c.b)("em",{parentName:"a"},"CameraPosition"))," ; ",Object(c.b)("inlineCode",{parentName:"p"},"supportsLowLightBoost"),": ",Object(c.b)("em",{parentName:"p"},"boolean"),"  }",">"),Object(c.b)("p",null,"Represents a camera device discovered by the ",Object(c.b)("inlineCode",{parentName:"p"},"Camera.getAvailableCameraDevices()")," function"),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L159"},"CameraDevice.ts:159")),Object(c.b)("hr",null),Object(c.b)("h3",{id:"cameradeviceformat"},"CameraDeviceFormat"),Object(c.b)("p",null,"\u01ac ",Object(c.b)("strong",{parentName:"p"},"CameraDeviceFormat"),": ",Object(c.b)("em",{parentName:"p"},"Readonly"),"<{ ",Object(c.b)("inlineCode",{parentName:"p"},"autoFocusSystem"),": ",Object(c.b)("a",{parentName:"p",href:"/react-native-vision-camera/docs/api/modules/cameradevice#autofocussystem"},Object(c.b)("em",{parentName:"a"},"AutoFocusSystem"))," ; ",Object(c.b)("inlineCode",{parentName:"p"},"colorSpaces"),": ",Object(c.b)("a",{parentName:"p",href:"/react-native-vision-camera/docs/api/modules/cameradevice#colorspace"},Object(c.b)("em",{parentName:"a"},"ColorSpace")),"[] ; ",Object(c.b)("inlineCode",{parentName:"p"},"fieldOfView"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"frameRateRanges"),": ",Object(c.b)("a",{parentName:"p",href:"/react-native-vision-camera/docs/api/modules/cameradevice#frameraterange"},Object(c.b)("em",{parentName:"a"},"FrameRateRange")),"[] ; ",Object(c.b)("inlineCode",{parentName:"p"},"isHighestPhotoQualitySupported?"),": ",Object(c.b)("em",{parentName:"p"},"boolean")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"maxISO"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"maxZoom"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"minISO"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"photoHeight"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"photoWidth"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"supportsPhotoHDR"),": ",Object(c.b)("em",{parentName:"p"},"boolean")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"supportsVideoHDR"),": ",Object(c.b)("em",{parentName:"p"},"boolean")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"videoHeight?"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"videoStabilizationModes"),": ",Object(c.b)("a",{parentName:"p",href:"/react-native-vision-camera/docs/api/modules/cameradevice#videostabilizationmode"},Object(c.b)("em",{parentName:"a"},"VideoStabilizationMode")),"[] ; ",Object(c.b)("inlineCode",{parentName:"p"},"videoWidth?"),": ",Object(c.b)("em",{parentName:"p"},"number"),"  }",">"),Object(c.b)("p",null,"A Camera Device's video format. Do not create instances of this type yourself, only use ",Object(c.b)("inlineCode",{parentName:"p"},"Camera.getAvailableCameraDevices(...)"),"."),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L85"},"CameraDevice.ts:85")),Object(c.b)("hr",null),Object(c.b)("h3",{id:"colorspace"},"ColorSpace"),Object(c.b)("p",null,"\u01ac ",Object(c.b)("strong",{parentName:"p"},"ColorSpace"),": ",Object(c.b)("em",{parentName:"p"},"hlg-bt2020")," ","|"," ",Object(c.b)("em",{parentName:"p"},"p3-d65")," ","|"," ",Object(c.b)("em",{parentName:"p"},"srgb")," ","|"," ",Object(c.b)("em",{parentName:"p"},"yuv")),Object(c.b)("p",null,"Indicates a format's color space."),Object(c.b)("h4",{id:"the-following-colorspaces-are-available-on-ios"},"The following colorspaces are available on iOS:"),Object(c.b)("ul",null,Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"srgb"'),": The sGRB color space (",Object(c.b)("a",{parentName:"li",href:"https://www.w3.org/Graphics/Color/srgb"},"https://www.w3.org/Graphics/Color/srgb"),")"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"p3-d65"'),": The P3 D65 wide color space which uses Illuminant D65 as the white point"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"hlg-bt2020"'),": The BT2020 wide color space which uses Illuminant D65 as the white point and Hybrid Log-Gamma as the transfer function")),Object(c.b)("h4",{id:"the-following-colorspaces-are-available-on-android"},"The following colorspaces are available on Android:"),Object(c.b)("ul",null,Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"yuv"'),": The YCbCr color space.")),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L55"},"CameraDevice.ts:55")),Object(c.b)("hr",null),Object(c.b)("h3",{id:"frameraterange"},"FrameRateRange"),Object(c.b)("p",null,"\u01ac ",Object(c.b)("strong",{parentName:"p"},"FrameRateRange"),": ",Object(c.b)("em",{parentName:"p"},"Readonly"),"<{ ",Object(c.b)("inlineCode",{parentName:"p"},"maxFrameRate"),": ",Object(c.b)("em",{parentName:"p"},"number")," ; ",Object(c.b)("inlineCode",{parentName:"p"},"minFrameRate"),": ",Object(c.b)("em",{parentName:"p"},"number"),"  }",">"),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L77"},"CameraDevice.ts:77")),Object(c.b)("hr",null),Object(c.b)("h3",{id:"logicalcameradevicetype"},"LogicalCameraDeviceType"),Object(c.b)("p",null,"\u01ac ",Object(c.b)("strong",{parentName:"p"},"LogicalCameraDeviceType"),": ",Object(c.b)("em",{parentName:"p"},"dual-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"dual-wide-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"triple-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"true-depth-camera")),Object(c.b)("p",null,"Indentifiers for a logical camera (Combinations of multiple physical cameras to create a single logical camera)."),Object(c.b)("ul",null,Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"dual-camera"'),": A combination of wide-angle and telephoto cameras that creates a capture device."),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"dual-wide-camera"'),": A device that consists of two cameras of fixed focal length, one ultrawide angle and one wide angle."),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"triple-camera"'),": A device that consists of three cameras of fixed focal length, one ultrawide angle, one wide angle, and one telephoto."),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"true-depth-camera"'),": A combination of cameras and other sensors that creates a capture device capable of photo, video, and depth capture.")),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L20"},"CameraDevice.ts:20")),Object(c.b)("hr",null),Object(c.b)("h3",{id:"physicalcameradevicetype"},"PhysicalCameraDeviceType"),Object(c.b)("p",null,"\u01ac ",Object(c.b)("strong",{parentName:"p"},"PhysicalCameraDeviceType"),": ",Object(c.b)("em",{parentName:"p"},"ultra-wide-angle-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"wide-angle-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"telephoto-camera")),Object(c.b)("p",null,"Indentifiers for a physical camera (one that actually exists on the back/front of the device)"),Object(c.b)("ul",null,Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"ultra-wide-angle-camera"'),": A built-in camera with a shorter focal length than that of a wide-angle camera. (focal length between below 24mm)"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"wide-angle-camera"'),": A built-in wide-angle camera. (focal length between 24mm and 35mm)"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"telephoto-camera"'),": A built-in camera device with a longer focal length than a wide-angle camera. (focal length between above 85mm)")),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L10"},"CameraDevice.ts:10")),Object(c.b)("hr",null),Object(c.b)("h3",{id:"videostabilizationmode"},"VideoStabilizationMode"),Object(c.b)("p",null,"\u01ac ",Object(c.b)("strong",{parentName:"p"},"VideoStabilizationMode"),": ",Object(c.b)("em",{parentName:"p"},"off")," ","|"," ",Object(c.b)("em",{parentName:"p"},"standard")," ","|"," ",Object(c.b)("em",{parentName:"p"},"cinematic")," ","|"," ",Object(c.b)("em",{parentName:"p"},"cinematic-extended")," ","|"," ",Object(c.b)("em",{parentName:"p"},"auto")),Object(c.b)("p",null,"Indicates a format's supported video stabilization mode"),Object(c.b)("ul",null,Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"off"'),": Indicates that video should not be stabilized"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"standard"'),": Indicates that video should be stabilized using the standard video stabilization algorithm introduced with iOS 5.0. Standard video stabilization has a reduced field of view. Enabling video stabilization may introduce additional latency into the video capture pipeline"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"cinematic"'),": Indicates that video should be stabilized using the cinematic stabilization algorithm for more dramatic results. Cinematic video stabilization has a reduced field of view compared to standard video stabilization. Enabling cinematic video stabilization introduces much more latency into the video capture pipeline than standard video stabilization and consumes significantly more system memory. Use narrow or identical min and max frame durations in conjunction with this mode"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"cinematic-extended"'),": Indicates that the video should be stabilized using the extended cinematic stabilization algorithm. Enabling extended cinematic stabilization introduces longer latency into the video capture pipeline compared to the AVCaptureVideoStabilizationModeCinematic and consumes more memory, but yields improved stability. It is recommended to use identical or similar min and max frame durations in conjunction with this mode (iOS 13.0+)"),Object(c.b)("li",{parentName:"ul"},Object(c.b)("inlineCode",{parentName:"li"},'"auto"'),": Indicates that the most appropriate video stabilization mode for the device and format should be chosen automatically")),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L75"},"CameraDevice.ts:75")),Object(c.b)("h2",{id:"functions"},"Functions"),Object(c.b)("h3",{id:"parsephysicaldevicetypes"},"parsePhysicalDeviceTypes"),Object(c.b)("p",null,"\u25b8 ",Object(c.b)("inlineCode",{parentName:"p"},"Const"),Object(c.b)("strong",{parentName:"p"},"parsePhysicalDeviceTypes"),"(",Object(c.b)("inlineCode",{parentName:"p"},"physicalDeviceTypes"),": ",Object(c.b)("a",{parentName:"p",href:"/react-native-vision-camera/docs/api/modules/cameradevice#physicalcameradevicetype"},Object(c.b)("em",{parentName:"a"},"PhysicalCameraDeviceType")),"[]): ",Object(c.b)("em",{parentName:"p"},"ultra-wide-angle-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"wide-angle-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"telephoto-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"dual-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"dual-wide-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"triple-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"true-depth-camera")),Object(c.b)("p",null,"Parses an array of physical device types into a single ",Object(c.b)("inlineCode",{parentName:"p"},"PhysicalCameraDeviceType")," or ",Object(c.b)("inlineCode",{parentName:"p"},"LogicalCameraDeviceType"),", depending what matches."),Object(c.b)("p",null,Object(c.b)("strong",{parentName:"p"},Object(c.b)("inlineCode",{parentName:"strong"},"method"))," "),Object(c.b)("h4",{id:"parameters"},"Parameters:"),Object(c.b)("table",null,Object(c.b)("thead",{parentName:"table"},Object(c.b)("tr",{parentName:"thead"},Object(c.b)("th",{parentName:"tr",align:"left"},"Name"),Object(c.b)("th",{parentName:"tr",align:"left"},"Type"))),Object(c.b)("tbody",{parentName:"table"},Object(c.b)("tr",{parentName:"tbody"},Object(c.b)("td",{parentName:"tr",align:"left"},Object(c.b)("inlineCode",{parentName:"td"},"physicalDeviceTypes")),Object(c.b)("td",{parentName:"tr",align:"left"},Object(c.b)("a",{parentName:"td",href:"/react-native-vision-camera/docs/api/modules/cameradevice#physicalcameradevicetype"},Object(c.b)("em",{parentName:"a"},"PhysicalCameraDeviceType")),"[]")))),Object(c.b)("p",null,Object(c.b)("strong",{parentName:"p"},"Returns:")," ",Object(c.b)("em",{parentName:"p"},"ultra-wide-angle-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"wide-angle-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"telephoto-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"dual-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"dual-wide-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"triple-camera")," ","|"," ",Object(c.b)("em",{parentName:"p"},"true-depth-camera")),Object(c.b)("p",null,"Defined in: ",Object(c.b)("a",{parentName:"p",href:"https://github.com/cuvent/react-native-vision-camera/blob/2925b84/src/CameraDevice.ts#L26"},"CameraDevice.ts:26")))}l.isMDXComponent=!0}}]);