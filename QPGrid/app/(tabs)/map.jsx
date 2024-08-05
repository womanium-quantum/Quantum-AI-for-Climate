import { StyleSheet, Text, View } from 'react-native'
import React from 'react'

const MapScreen = () => {
    return (
        <View style={styles.container}>
            <Text>This is the Map Screen</Text>
        </View>
    );
};

export default MapScreen;

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#fff',
    },
});

