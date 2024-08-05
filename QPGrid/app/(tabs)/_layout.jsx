import { View, Text } from 'react-native'
import {Tabs, Redirect} from 'expo-router';

const TabsLayout = () => {
    return (
        <Tabs>
            {/* Home Screen */}
            <Tabs.Screen
            name="home"
            options={{
                title: 'Home',
                tabBarIcon: ({ color, size }) => (
                <Text style={{ fontSize: size, color }}>🏠</Text>
                ),
            }}
            />
            {/* Map Screen */}
            <Tabs.Screen
            name="map"
            options={{
                title: 'Map',
                tabBarIcon: ({ color, size }) => (
                <Text style={{ fontSize: size, color }}>🗺️</Text>
                ),
            }}
            />
            {/* Info Screen */}
            <Tabs.Screen
            name="info"
            options={{
                title: 'Info',
                tabBarIcon: ({ color, size }) => (
                <Text style={{ fontSize: size, color }}>ℹ️</Text>
                ),
            }}
            />
        </Tabs>
    );
};

export default TabsLayout;