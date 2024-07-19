"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = exports.NativeSearchBarCommands = exports.NativeSearchBar = void 0;
var _react = _interopRequireDefault(require("react"));
var _reactNativeScreens = require("react-native-screens");
var _reactNative = require("react-native");
var _SearchBarNativeComponent = _interopRequireWildcard(require("../fabric/SearchBarNativeComponent"));
function _getRequireWildcardCache(e) { if ("function" != typeof WeakMap) return null; var r = new WeakMap(), t = new WeakMap(); return (_getRequireWildcardCache = function (e) { return e ? t : r; })(e); }
function _interopRequireWildcard(e, r) { if (!r && e && e.__esModule) return e; if (null === e || "object" != typeof e && "function" != typeof e) return { default: e }; var t = _getRequireWildcardCache(r); if (t && t.has(e)) return t.get(e); var n = { __proto__: null }, a = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var u in e) if ("default" !== u && Object.prototype.hasOwnProperty.call(e, u)) { var i = a ? Object.getOwnPropertyDescriptor(e, u) : null; i && (i.get || i.set) ? Object.defineProperty(n, u, i) : n[u] = e[u]; } return n.default = e, t && t.set(e, n), n; }
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _extends() { _extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); } // Native components
const NativeSearchBar = exports.NativeSearchBar = _SearchBarNativeComponent.default;
const NativeSearchBarCommands = exports.NativeSearchBarCommands = _SearchBarNativeComponent.Commands;
function SearchBar(props, ref) {
  const searchBarRef = _react.default.useRef(null);
  _react.default.useImperativeHandle(ref, () => ({
    blur: () => {
      _callMethodWithRef(ref => NativeSearchBarCommands.blur(ref));
    },
    focus: () => {
      _callMethodWithRef(ref => NativeSearchBarCommands.focus(ref));
    },
    toggleCancelButton: flag => {
      _callMethodWithRef(ref => NativeSearchBarCommands.toggleCancelButton(ref, flag));
    },
    clearText: () => {
      _callMethodWithRef(ref => NativeSearchBarCommands.clearText(ref));
    },
    setText: text => {
      _callMethodWithRef(ref => NativeSearchBarCommands.setText(ref, text));
    },
    cancelSearch: () => {
      _callMethodWithRef(ref => NativeSearchBarCommands.cancelSearch(ref));
    }
  }));
  const _callMethodWithRef = _react.default.useCallback(method => {
    const ref = searchBarRef.current;
    if (ref) {
      method(ref);
    } else {
      console.warn('Reference to native search bar component has not been updated yet');
    }
  }, [searchBarRef]);
  if (!_reactNativeScreens.isSearchBarAvailableForCurrentPlatform) {
    console.warn('Importing SearchBar is only valid on iOS and Android devices.');
    return _reactNative.View;
  }
  return /*#__PURE__*/_react.default.createElement(NativeSearchBar, _extends({
    ref: searchBarRef
  }, props, {
    onSearchFocus: props.onFocus,
    onSearchBlur: props.onBlur,
    onSearchButtonPress: props.onSearchButtonPress,
    onCancelButtonPress: props.onCancelButtonPress,
    onChangeText: props.onChangeText
  }));
}
var _default = exports.default = /*#__PURE__*/_react.default.forwardRef(SearchBar);
//# sourceMappingURL=SearchBar.js.map